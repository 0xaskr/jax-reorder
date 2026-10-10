"""Lossless in-memory values for structural comparison, never text summaries."""

import dataclasses
import enum
import struct
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from jax._src import core, flattree, layout, mesh, named_sharding
from jax._src.hashable_array import HashableArray
from jax._src.lax.lax import NoDCEEffect
from jax._src.numpy.scalar_types import _ScalarMeta
from jax._src.random.prng import KeyTy, PRNGKeyArray

from .errors import UnsupportedJaxprError

type Frozen = Any


class Encoder:
    def __init__(self, nested: Callable, variable: Callable):
        self.nested = nested
        self.variable = variable
        self.active: set[int] = set()

    def fields(self, obj: Any, names: tuple[str, ...], path: str) -> Frozen:
        return (
            type(obj),
            tuple((name, self.encode(getattr(obj, name), f"{path}.{name}")) for name in names),
        )

    def literal(self, value: Any, aval: core.AbstractValue, path: str) -> Frozen:
        # Literal storage may be a Python scalar, TypedFloat/TypedInt, or NumPy
        # scalar. Its meaning is determined by its declared dtype, not storage.
        if isinstance(aval, core.ShapedArray):
            return self.encode(np.asarray(value, dtype=aval.dtype), path)
        return self.encode(value, path)

    def encode(self, value: Any, path: str) -> Frozen:
        if id(value) in self.active:
            raise UnsupportedJaxprError(path, "cyclic parameter container")
        self.active.add(id(value))
        try:
            return self._encode(value, path)
        finally:
            self.active.remove(id(value))

    def _encode(self, value: Any, path: str) -> Frozen:
        if isinstance(value, core.Jaxpr):
            return ("jaxpr", self.nested(value, path).graph)
        if isinstance(value, core.Var):
            return ("variable", self.variable(value))
        if isinstance(value, core.Primitive):
            return ("primitive", value)
        if isinstance(value, NoDCEEffect):
            return ("no_dce_effect",)
        if isinstance(value, HashableArray):
            return self.fields(value, ("val",), path)
        if isinstance(value, _ScalarMeta) or (
            isinstance(value, type)
            and (issubclass(value, np.generic) or value in (bool, int, float, complex))
        ):
            return ("dtype", np.dtype(value))
        if value is None:
            return ("none",)
        if isinstance(value, enum.Enum):
            return ("enum", type(value), self.encode(value.value, path))
        if type(value) in (bool, int, str, bytes):
            return (type(value), value)
        if type(value) is float:
            return (float, struct.pack(">d", value))
        if type(value) is complex:
            return (complex, struct.pack(">dd", value.real, value.imag))
        if isinstance(value, np.dtype):
            # dtype.str alone conflates distinct ml_dtypes (e.g. float8 formats).
            return ("dtype", value)
        if isinstance(value, KeyTy):
            # KeyTy equality compares its complete PRNGImpl, including function
            # identities. It is an audited immutable descriptor, not its name.
            return ("key_dtype", value)
        if isinstance(value, PRNGKeyArray):
            return (
                "key_array",
                self.encode(value.dtype, path),
                self.encode(jax.random.key_data(value), path),
            )
        if isinstance(value, (np.ndarray, np.generic, jax.Array)):
            if isinstance(value, jax.Array) and any(d.platform != "cpu" for d in value.devices()):
                raise UnsupportedJaxprError(path, "only CPU arrays are currently supported")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise UnsupportedJaxprError(path, "object arrays have opaque element semantics")
            return ("array", array.dtype, array.shape, array.tobytes(order="C"))
        if isinstance(value, core.AbstractToken):
            return ("token",)
        if isinstance(value, jax.tree_util.PyTreeDef):
            # Auxiliary data may itself contain opaque Python objects. Encode
            # it explicitly instead of delegating to arbitrary __eq__ methods.
            node = value.node_data()
            if node is None:
                return ("pytree_leaf",)
            node_type, metadata = node
            return (
                "pytree",
                node_type,
                self.encode(metadata, f"{path}.metadata"),
                tuple(
                    self.encode(c, f"{path}.children[{i}]") for i, c in enumerate(value.children())
                ),
            )
        flat_fields = {
            flattree.FTTuple: ("elts",),
            flattree.FTDict: ("keys", "_vals"),
            flattree.FTPyTree: ("xs", "_tree"),
            flattree.FTSingleton: ("val",),
            flattree.FTStatic: ("val",),
            flattree.FTFiltered: ("val",),
        }
        if type(value) in flat_fields:
            return self.fields(value, flat_fields[type(value)], path)
        if isinstance(value, core.ShapedArray):
            # _stripped_weak_type is a derived cache, not another semantic field.
            return self.fields(
                value,
                (
                    "shape",
                    "dtype",
                    "weak_type",
                    "sharding",
                    "manual_axis_type",
                    "memory_space",
                    "layout",
                ),
                path,
            )
        if isinstance(value, core.ManualAxisType):
            return self.fields(value, ("varying", "unreduced", "reduced", "unreduced_kind"), path)
        if isinstance(value, core.JaxprEqnContext):
            names = tuple(
                name for name in value.__slots__ if name != "__weakref__" and hasattr(value, name)
            )
            return self.fields(value, names, path)
        if isinstance(value, mesh.AbstractMesh):
            return self.fields(
                value, ("axis_sizes", "axis_names", "axis_types", "abstract_device"), path
            )
        if isinstance(value, mesh.Mesh):
            devices = tuple(self.encode(v, f"{path}.devices") for v in value.devices.flat)
            return (
                "mesh",
                value.devices.shape,
                devices,
                self.encode(value.axis_names, path),
                self.encode(value.axis_types, path),
            )
        if isinstance(value, jax.Device):
            if value.platform != "cpu":
                raise UnsupportedJaxprError(path, "only CPU device placement is supported")
            return ("device", value.platform, value.process_index, value.id, value.device_kind)
        if isinstance(value, jax.sharding.NamedSharding):
            return self.fields(value, ("mesh", "spec", "memory_kind", "_logical_device_ids"), path)
        if isinstance(value, jax.sharding.SingleDeviceSharding):
            return self.fields(value, ("_device", "memory_kind"), path)
        if isinstance(value, jax.sharding.PartitionSpec):
            return self.fields(
                value, ("partitions", "unreduced", "reduced", "unreduced_kind"), path
            )
        if value is jax.sharding.PartitionSpec.UNCONSTRAINED:
            return ("unconstrained_partition",)
        if value is layout.AutoLayout or value is named_sharding.UNSPECIFIED:
            return ("singleton", type(value))
        if isinstance(value, layout.Layout):
            return self.fields(
                value, ("major_to_minor", "tiling", "sub_byte_element_size_in_bits"), path
            )
        if isinstance(value, slice):
            return (slice, self.encode((value.start, value.stop, value.step), path))
        if isinstance(value, range):
            return (range, value.start, value.stop, value.step)
        if isinstance(value, tuple):
            return (type(value), tuple(self.encode(v, f"{path}[{i}]") for i, v in enumerate(value)))
        if type(value) is list:
            return (list, tuple(self.encode(v, f"{path}[{i}]") for i, v in enumerate(value)))
        if type(value) is dict:
            return (
                dict,
                frozenset(
                    (self.encode(k, path), self.encode(v, f"{path}[{k!r}]"))
                    for k, v in value.items()
                ),
            )
        if type(value) in (set, frozenset):
            return (type(value), frozenset(self.encode(v, path) for v in value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            if type(value).__module__.startswith(("jax.", "jaxlib.")):
                return self.fields(value, tuple(f.name for f in dataclasses.fields(value)), path)
        raise UnsupportedJaxprError(
            path,
            "opaque parameter or abstract value of type "
            f"{type(value).__module__}.{type(value).__qualname__}",
        )


def replace_nested(value: Any, replace: Callable) -> Any:
    """Replace sub-jaxprs while retaining container types and all other values."""
    if isinstance(value, core.Jaxpr):
        return replace(value)
    if isinstance(value, tuple):
        new = tuple(replace_nested(v, replace) for v in value)
        if all(a is b for a, b in zip(value, new, strict=True)):
            return value
        return type(value)(*new) if hasattr(value, "_fields") else type(value)(new)
    if type(value) is list:
        new = [replace_nested(v, replace) for v in value]
        return value if all(a is b for a, b in zip(value, new, strict=True)) else new
    if type(value) is dict:
        new = {k: replace_nested(v, replace) for k, v in value.items()}
        return value if all(value[k] is v for k, v in new.items()) else new
    return value
