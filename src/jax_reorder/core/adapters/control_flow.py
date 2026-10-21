"""Decode bounded-loop interfaces without evaluating or erasing their effects."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from jax import lax
from jax._src import core, flattree

from ..model import (
    ArrayType,
    AxisEffectView,
    Effect,
    EffectView,
    GlobalEffectView,
    Operand,
    ParameterView,
    RegionView,
    ValueRef,
)
from .attributes import attribute, require

if TYPE_CHECKING:
    from .jax import JaxAdapter

type LoopRegion = tuple[str, RegionView, tuple[Operand, ...]]


def adapt_loop(
    eq: Any, adapter: JaxAdapter, args: tuple[Operand, ...], effects: tuple[Effect, ...], path: str
) -> tuple[tuple[ParameterView, ...], tuple[LoopRegion, ...]]:
    p = eq.params
    input_types = tuple(adapter.value_type(v.aval, path) for v in eq.invars)
    output_types = tuple(adapter.value_type(v.aval, path) for v in eq.outvars)
    if eq.primitive is lax.scan_p:
        require(
            set(p) == {"reverse", "length", "jaxpr", "ft_in", "ft_out", "unroll"},
            path,
            "unaudited scan schema",
        )
        require(
            type(p["length"]) is int
            and p["length"] >= 0
            and type(p["unroll"]) is int
            and p["unroll"] >= 0
            and type(p["reverse"]) is bool
            and type(p["ft_in"]) is flattree.FTTuple
            and type(p["ft_out"]) is flattree.FTTuple,
            path,
            "invalid scan metadata",
        )
        groups = tuple(tuple(g) for g in p["ft_in"].update(range(len(args))).unpack())
        out_groups = tuple(tuple(g) for g in p["ft_out"].update(range(len(output_types))).unpack())
        require(len(groups) == 3 and len(out_groups) == 2, path, "invalid scan port groups")
        require(
            tuple(i for g in groups for i in g) == tuple(range(len(args)))
            and tuple(i for g in out_groups for i in g) == tuple(range(len(output_types))),
            path,
            "scan port groups are not ordered partitions",
        )
        body = adapter.region(p["jaxpr"], path + ".body")
        expected = list(input_types)
        for slot in groups[2]:
            typ = expected[slot]
            require(isinstance(typ, ArrayType) and bool(typ.shape), path, "invalid scan sequence")
            assert isinstance(typ, ArrayType)
            require(typ.shape[0] == p["length"], path, "scan length/sequence mismatch")
            expected[slot] = replace(typ, shape=typ.shape[1:])
        # JAX's checker validates sliced sequence metadata; bind the exact
        # child types for sequence slots, whose rank-specific sharding changes.
        for i, (var, typ) in enumerate(zip(body.inputs, expected, strict=True)):
            if i in groups[2]:
                require(
                    isinstance(var.type, ArrayType)
                    and isinstance(typ, ArrayType)
                    and (var.type.shape, var.type.dtype) == (typ.shape, typ.dtype),
                    path,
                    "scan sequence binding differs",
                )
            else:
                require(var.type == typ, path, "scan constant/carry binding differs")
        require(
            all(
                core.typematch(eq.invars[i].aval, eq.outvars[o].aval)
                for i, o in zip(groups[1], out_groups[0], strict=True)
            ),
            path,
            "scan carry types differ",
        )
        params = tuple(
            ParameterView(k, attribute(v, path) if k.startswith("ft_") else v)
            for k, v in sorted(p.items())
            if k != "jaxpr"
        ) + (ParameterView("input_groups", groups), ParameterView("output_groups", out_groups))
        regions = (("body", body, args),)
    else:
        require(
            set(p) == {"cond_nconsts", "cond_jaxpr", "body_nconsts", "body_jaxpr"},
            path,
            "unaudited while schema",
        )
        nc, nb = p["cond_nconsts"], p["body_nconsts"]
        require(
            type(nc) is int and type(nb) is int and nc >= 0 and nb >= 0 and nc + nb <= len(args),
            path,
            "invalid while constant groups",
        )
        cond = adapter.region(p["cond_jaxpr"], path + ".condition")
        body = adapter.region(p["body_jaxpr"], path + ".body")
        require(
            tuple(v.type for v in cond.inputs) == input_types[:nc] + input_types[nc + nb :]
            and tuple(v.type for v in body.inputs) == input_types[nc:]
            and output_types == input_types[nc + nb :],
            path,
            "while port types differ",
        )
        params = (ParameterView("cond_nconsts", nc), ParameterView("body_nconsts", nb))
        regions = (("condition", cond, args[:nc] + args[nc + nb :]), ("body", body, args[nc:]))
    mapped: set[Effect] = set()
    for _, region, operands in regions:
        require(not region.is_high, path, "high-level loop region")
        bindings = dict(zip((v.id for v in region.inputs), operands, strict=True))
        for effect in region.effects:
            if isinstance(effect, AxisEffectView | GlobalEffectView):
                mapped.add(effect)
            else:
                ref = bindings.get(effect.ref.id)
                require(isinstance(ref, ValueRef), path, "loop effect on captured Ref")
                assert isinstance(ref, ValueRef)
                mapped.add(EffectView(effect.kind, ref))
    require(set(effects) == mapped, path, "loop effects do not bind to operands")
    return params, regions
