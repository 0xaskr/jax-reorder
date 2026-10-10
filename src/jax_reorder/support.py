"""Primitive identities, not names or a dtype/shape sample whitelist.

Discovery includes implementation primitives used by public lax wrappers. The
coverage tests separately account for the exported inventory and traced helpers.
Membership alone does not permit dropping parameters or effects.
"""

import sys

from jax import lax
from jax._src import pjit, shard_map
from jax.extend import core


def public_primitives() -> dict[core.Primitive, tuple[str, ...]]:
    names: dict[core.Primitive, list[str]] = {}
    for module in (lax, lax.linalg):
        for name, value in vars(module).items():
            if not name.startswith("_") and isinstance(value, core.Primitive):
                names.setdefault(value, []).append(f"{module.__name__}.{name}")
    return {primitive: tuple(sorted(paths)) for primitive, paths in names.items()}


def supported_primitive_identities() -> frozenset[core.Primitive]:
    result = set(public_primitives())
    # These modules define the primitives to which public lax calls are traced.
    for name, module in tuple(sys.modules.items()):
        if module is not None and name.startswith("jax._src.lax."):
            result.update(v for v in vars(module).values() if isinstance(v, core.Primitive))
    result.update(v for v in vars(pjit).values() if isinstance(v, core.Primitive))
    # This separate JAX API encodes instruction scheduling inside its body;
    # it is not produced by the supported public lax wrappers.
    result.discard(pjit.program_order_p)
    result.add(shard_map.shard_map_p)
    return frozenset(result)
