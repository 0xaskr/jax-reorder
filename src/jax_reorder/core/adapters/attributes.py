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
