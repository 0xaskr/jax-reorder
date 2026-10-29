"""Region-local SSA consumption, including effect and nested-call operands."""

from dataclasses import fields, is_dataclass
from typing import Any

from .model import RegionView, ValueRef


def consumed_values(region: RegionView) -> frozenset[int]:
    # Do not traverse child bodies: their SSA namespaces are independent.
    # Explicit bindings capture every use of this region's values by a child.
    pending: list[Any] = list(region.outputs)
    for op in region.operations:
        pending.extend(op.operands)
        pending.extend((op.access, op.dma, op.semaphore))
        for binding in op.regions:
            pending.extend(binding.arguments)
    used: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, ValueRef):
            used.add(value.id)
        elif isinstance(value, tuple):
            pending.extend(value)
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in fields(value))
    return frozenset(used)
