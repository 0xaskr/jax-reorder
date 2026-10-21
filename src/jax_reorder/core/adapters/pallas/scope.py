"""Bind run_scoped captures separately from fresh, nonescaping local resources."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jax._src import core
from jax._src.pallas.mosaic import core as mc
from jax._src.state import types as st

from ...model import (
    ArrayType,
    Effect,
    EffectView,
    Operand,
    RefType,
    RegionView,
    ScratchView,
    SemaphoreType,
    ValueRef,
)
from ..attributes import require

if TYPE_CHECKING:
    from ..jax import JaxAdapter


def adapt_scope(
    eq: Any, adapter: JaxAdapter, args: tuple[Operand, ...], effects: tuple[Effect, ...], path: str
) -> tuple[RegionView, tuple[ScratchView, ...]]:
    p = eq.params
    require(
        set(p) == {"jaxpr", "collective_axes", "ref_transforms"},
        path,
        "unaudited run_scoped schema",
    )
    source = p["jaxpr"]
    require(type(source) is core.Jaxpr and not source.is_high, path, "unsupported scoped region")
    require(
        p["collective_axes"] == () and p["ref_transforms"] == ((),) * len(source.invars),
        path,
        "scoped collective allocations/transforms need a lifetime/layout proof",
    )
    require(
        len(args) == len(source.constvars)
        and tuple(adapter.value_type(v.aval, path) for v in eq.invars)
        == tuple(adapter.value_type(v.aval, path) for v in source.constvars),
        path,
        "scoped capture bindings differ",
    )
    require(
        tuple(adapter.value_type(v.aval, path) for v in source.outvars)
        == tuple(adapter.value_type(v.aval, path) for v in eq.outvars)
        and all(isinstance(adapter.value_type(v.aval, path), ArrayType) for v in source.outvars),
        path,
        "scoped results must be arrays; Ref escape is not supported",
    )
    allocations = []
    for var in source.invars:
        require(type(var.aval) is st.AbstractRef, path, "scoped allocation must be an AbstractRef")
        assert isinstance(var.aval, st.AbstractRef)
        typ = adapter.value_type(var.aval, path)
        require(isinstance(typ, RefType), path, "scoped allocation must be a Ref")
        assert isinstance(typ, RefType)
        semaphore = isinstance(typ.inner, SemaphoreType)
        require(
            semaphore or var.aval.memory_space is mc.MemorySpace.VMEM,
            path,
            "only VMEM and semaphore scoped allocations are reviewed",
        )
        require(var.aval.kind is None, path, "unknown scoped Ref kind")
        allocations.append(
            ScratchView(
                typ,
                "scope",
                0 if semaphore else None,
                typ.inner if isinstance(typ.inner, ArrayType) else None,
            )
        )
    child = adapter.region(
        source.replace(constvars=(), invars=(*source.constvars, *source.invars)), path + ".scope"
    )
    expected: set[Effect] = set()
    bindings = dict(zip((v.id for v in child.inputs[: len(args)]), args, strict=True))
    for effect in child.effects:
        if isinstance(effect, EffectView):
            if effect.ref.id not in bindings:
                continue
            ref = bindings[effect.ref.id]
            require(isinstance(ref, ValueRef), path, "scoped effect has no outer Ref")
            assert isinstance(ref, ValueRef)
            expected.add(EffectView(effect.kind, ref))
        else:
            expected.add(effect)
    require(set(effects) == expected, path, "scoped effects do not bind to outer captures")
    return child, tuple(allocations)
