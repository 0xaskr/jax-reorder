"""Portable exact attributes; source descriptors are decoded only by adapters."""

from __future__ import annotations

import struct
from typing import Any

from ...errors import UnsupportedJaxprError
from ..model import Attribute
from ..semantics import Encoder


def require(condition: bool, path: str, reason: str) -> None:
    if not condition:
        raise UnsupportedJaxprError(path, reason)


def _portable(value: Any, path: str) -> Attribute:
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if type(value) is float:
        return ("float64", struct.pack("!d", value))
    if isinstance(value, type):
        return ("type", value.__module__, value.__qualname__)
    if type(value) is tuple:
        return tuple(_portable(v, path) for v in value)
    if type(value) is frozenset:
        return ("set", tuple(sorted((_portable(v, path) for v in value), key=repr)))
    raise UnsupportedJaxprError(path, f"nonportable encoded descriptor {type(value).__name__}")


def attribute(value: Any, path: str) -> Attribute:
    def reject(*_: Any) -> Any:
        raise UnsupportedJaxprError(path, "region/variable must use a typed binding")

    return _portable(Encoder(reject, reject).encode(value, path), path)


def project_mesh_attribute(value: tuple[Any, ...], names: set[str]) -> tuple[Any, ...]:
    """Filter reviewed portable mesh axes, or preserve other source attributes."""
    if len(value) != 2 or value[0] != ("type", "jax._src.mesh", "AbstractMesh"):
        return value
    if not isinstance(value[1], tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 or type(item[0]) is not str
        for item in value[1]
    ):
        raise UnsupportedJaxprError("local.context", "unknown portable mesh fields")
    columns = dict(value[1])
    axis_fields = ("axis_names", "axis_sizes", "axis_types")
    tuple_tag = ("type", "builtins", "tuple")
    if (
        len(columns) != len(value[1])
        or set(columns) != {*axis_fields, "abstract_device"}
        or any(
            not isinstance(columns[name], tuple)
            or len(columns[name]) != 2
            or columns[name][0] != tuple_tag
            or not isinstance(columns[name][1], tuple)
            for name in axis_fields
        )
    ):
        raise UnsupportedJaxprError("local.context", "unknown portable mesh fields")
    labels = columns["axis_names"][1]
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or item[0] != ("type", "builtins", "str")
        or type(item[1]) is not str
        for item in labels
    ) or any(len(columns[name][1]) != len(labels) for name in axis_fields):
        raise UnsupportedJaxprError("local.context", "invalid portable mesh axis columns")
    selected = tuple(i for i, item in enumerate(labels) if item[1] in names)
    return (
        value[0],
        tuple(
            (
                name,
                (tuple_tag, tuple(item[1][i] for i in selected))
                if name in axis_fields
                else item
                if selected
                else ("none",),
            )
            for name, item in value[1]
        ),
    )
