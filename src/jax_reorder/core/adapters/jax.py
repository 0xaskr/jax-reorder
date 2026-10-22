"""Adapt the audited straight-line fragment once, preserving all source facts.

The legacy JAX consumers migrate separately. This adapter is intentionally
fail-closed: recognizing a primitive identity does not license unknown schemas.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import jax
import jaxlib
import numpy as np
from jax import lax
from jax._src import core, custom_derivatives, pjit, shard_map
from jax._src.api import Inline
from jax._src.lax import parallel
from jax._src.pallas import core as pc
from jax._src.pallas import primitives as pp
from jax._src.pallas.mosaic import core as mc
from jax._src.pallas.pallas_call import pallas_call_p
from jax._src.sharding_impls import UNSPECIFIED
from jax._src.state import primitives as sp
from jax._src.state import types as st

from ...errors import (
    ComparisonBudgetError,
    ComparisonFailureKind,
    InvalidJaxprError,
)
from ..model import (
    AdaptationContext,
    AdaptedProgram,
    ArrayType,
    AxisEffectView,
    Effect,
    EffectView,
    GlobalEffectView,
    LiteralValue,
    Operand,
    OperationView,
    Origin,
    ParameterView,
    ProgramView,
    RefType,
    RegionBinding,
    RegionView,
    SemaphoreType,
    SourceHandle,
    ValueDef,
    ValueRef,
    ValueType,
)
from ..params import semantic_parameters
from .attributes import attribute, require
from .control_flow import adapt_loop
from .pallas.call import adapt_call, adapt_memory_access
from .pallas.dma import DMA_SCHEMAS, adapt_dma
from .pallas.reshard import adapt_local_reshard, local_broadcast_sharding
from .pallas.scope import adapt_scope
from .pallas.semaphore import SEMAPHORE_SCHEMAS, adapt_semaphore
from .sharding import adapt_shard_map, sharding_view

# Explicit identities and parameter schemas; no name-based primitive acceptance.
PURE_SCHEMAS = {
    getattr(lax, f"{name}_p"): frozenset()
    for name in (
        "add",
        "sub",
        "mul",
        "div",
        "rem",
        "neg",
        "sign",
        "abs",
        "max",
        "min",
        "clamp",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "and",
        "or",
        "xor",
        "not",
        "select_n",
        "sin",
        "cos",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "tanh",
        "floor",
        "ceil",
        "is_finite",
        "stop_gradient",
    )
}
PURE_SCHEMAS.update(
    {
        lax.mul_p: frozenset({"out_dtype"}),
        **{
            getattr(lax, f"{name}_p"): frozenset({"accuracy"})
            for name in ("sin", "cos", "exp", "log", "sqrt", "rsqrt", "tanh")
        },
        lax.integer_pow_p: frozenset({"y"}),
        lax.convert_element_type_p: frozenset({"new_dtype", "weak_type", "sharding"}),
        lax.broadcast_in_dim_p: frozenset({"shape", "broadcast_dimensions", "sharding"}),
        lax.iota_p: frozenset({"dtype", "shape", "dimension", "sharding"}),
        lax.transpose_p: frozenset({"permutation"}),
        lax.reshape_p: frozenset({"new_sizes", "dimensions", "sharding"}),
        lax.slice_p: frozenset({"start_indices", "limit_indices", "strides"}),
        lax.concatenate_p: frozenset({"dimension"}),
        sp.broadcast_to_p: frozenset({"shape"}),
        lax.dot_general_p: frozenset(
            {"dimension_numbers", "precision", "preferred_element_type", "out_sharding"}
        ),
        lax.reduce_sum_p: frozenset({"axes", "out_sharding"}),
        lax.reduce_max_p: frozenset({"axes", "out_sharding"}),
        lax.reduce_min_p: frozenset({"axes", "out_sharding"}),
        lax.reduce_and_p: frozenset({"axes"}),
        lax.reduce_or_p: frozenset({"axes"}),
        lax.tile_p: frozenset({"reps"}),
        lax.gather_p: frozenset(
            {
                "dimension_numbers",
                "slice_sizes",
                "unique_indices",
                "indices_are_sorted",
                "mode",
                "fill_value",
            }
        ),
    }
)
MEMORY_SCHEMAS = {
    sp.get_p: frozenset({"tree"}),
    sp.swap_p: frozenset({"tree"}),
    pp.load_p: frozenset({"args_tree", "cache_modifier", "eviction_policy", "is_volatile"}),
    pp.swap_p: frozenset({"args_tree", "eviction_policy"}),
}
COORDINATE_SCHEMAS = {
    pp.program_id_p: frozenset({"axis"}),
    pp.num_programs_p: frozenset({"axis"}),
}
JIT_SCHEMA = frozenset(
    {
        "jaxpr",
        "name",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "donated_invars",
        "ctx_mesh",
        "keep_unused",
        "inline",
        "compiler_options_kvs",
    }
)


def parameter(name: str, value: Any, path: str) -> ParameterView:
    # These fields have a primitive-specific, type-checked schema. Give analysis
    # their semantic values rather than asking it to decode JAX descriptors.
    if name in {"axis", "dimension", "y", "weak_type", "unique_indices", "indices_are_sorted"}:
        require(type(value) in (int, bool), path, "unexpected scalar parameter")
        result = value
    elif name in ("dimensions", "strides") and value is None:
        result = ("none",)
    elif name in {
        "shape",
        "broadcast_dimensions",
        "new_sizes",
        "permutation",
        "axes",
        "reps",
        "slice_sizes",
        "dimensions",
        "start_indices",
        "limit_indices",
        "strides",
    }:
        require(
            type(value) is tuple and all(type(v) is int for v in value),
            path,
            "dynamic dimension parameter",
        )
        result = value
    elif name in {"dtype", "new_dtype"}:
        result = str(np.dtype(value))
    elif name == "preferred_element_type":
        result = ("none",) if value is None else str(np.dtype(value))
    elif name == "dimension_numbers" and type(value) is lax.GatherDimensionNumbers:
        require(
            value._fields
            == (
                "offset_dims",
                "collapsed_slice_dims",
                "start_index_map",
                "operand_batching_dims",
                "start_indices_batching_dims",
            ),
            path,
            "unaudited gather dimensions",
        )
        require(
            all(type(axes) is tuple and all(type(i) is int for i in axes) for axes in value),
            path,
            "invalid gather axes",
        )
        result = tuple(value)
    elif name == "mode" and type(value) is lax.GatherScatterMode:
        result = value.name
    elif name == "dimension_numbers" and type(value) is tuple:
        require(
            len(value) == 2
            and all(type(group) is tuple and len(group) == 2 for group in value)
            and all(
                type(axes) is tuple and all(type(i) is int for i in axes)
                for group in value
                for axes in group
            ),
            path,
            "invalid dot dimension numbers",
        )
        result = value
    else:
        result = attribute(value, path)
    return ParameterView(name, result)


class JaxAdapter:
    """One adaptation context; no global identity-only cache."""

    def __init__(self, context: AdaptationContext) -> None:
        self.context = context
        self.calls = 0
        self.in_pallas_kernel = False

    def kernel_region(self, source: Any, path: str) -> RegionView:
        previous = self.in_pallas_kernel
        self.in_pallas_kernel = True
        try:
            return self.region(source, path)
        finally:
            self.in_pallas_kernel = previous

    def value_type(self, aval: Any, path: str) -> ValueType:
        if type(aval) is st.AbstractRef:
            if type(aval.inner_aval) is core.ShapedArray and type(aval.dtype) in (
                mc.DMASemaphore,
                pc.Semaphore,
            ):
                inner_aval = aval.inner_aval
                kind = "dma" if type(aval.dtype) is mc.DMASemaphore else "regular"
                default = core.ShapedArray(inner_aval.shape, aval.dtype)
                require(
                    all(type(n) is int and n > 0 for n in inner_aval.shape)
                    and inner_aval.weak_type is False
                    and aval.memory_space is mc.MemorySpace.SEMAPHORE
                    and aval.kind is None,
                    path,
                    "only static semaphore arrays are adapted",
                )
                names = ("manual_axis_type", "memory_space", "layout", "sharding")
                metadata = tuple((n, attribute(getattr(inner_aval, n), path)) for n in names)
                require(
                    metadata == tuple((n, attribute(getattr(default, n), path)) for n in names),
                    path,
                    "non-default semaphore abstract contract",
                )
                return RefType(
                    SemaphoreType(inner_aval.shape, kind, metadata),
                    attribute(aval.memory_space, path),
                    attribute(aval.kind, path),
                )
            inner = self.value_type(aval.inner_aval, path + ".inner")
            require(isinstance(inner, ArrayType), path, "nested Ref")
            return RefType(
                cast(ArrayType, inner),
                attribute(aval.memory_space, path),
                attribute(aval.kind, path),
            )
        require(type(aval) is core.ShapedArray, path, "expected a static ShapedArray or Ref")
        require(all(type(d) is int and d >= 0 for d in aval.shape), path, "dynamic shape")
        require(
            getattr(aval.dtype, "kind", "") in ("b", "i", "u", "f", "c")
            or str(aval.dtype) == "bfloat16",
            path,
            "unsupported extended dtype",
        )
        default = core.ShapedArray(aval.shape, aval.dtype, weak_type=aval.weak_type)
        for name in ("memory_space", "layout"):
            require(
                attribute(getattr(aval, name), path) == attribute(getattr(default, name), path),
                path,
                f"non-default {name} is not yet admitted",
            )
        require(
            not aval.manual_axis_type.unreduced
            and not aval.manual_axis_type.reduced
            and aval.manual_axis_type.unreduced_kind is None,
            path,
            "reduction-state abstract values are not yet admitted",
        )
        return ArrayType(
            tuple(aval.shape),
            str(aval.dtype),
            bool(aval.weak_type),
            tuple(
                (name, attribute(getattr(aval, name), path + "." + name))
                for name in (
                    "manual_axis_type",
                    "memory_space",
                    "layout",
                )
            ),
            sharding_view(aval.sharding, path + ".sharding"),
        )

    def array_type(self, aval: Any, path: str) -> ArrayType:
        result = self.value_type(aval, path)
        require(isinstance(result, ArrayType), path, "expected array type")
        return cast(ArrayType, result)

    def literal(self, val: Any, aval: Any, path: str) -> LiteralValue:
        typ = self.array_type(aval, path)
        arr = np.asarray(val, dtype=aval.dtype)
        require(arr.shape == typ.shape, path, "literal shape differs from abstract value")
        return LiteralValue(typ, arr.tobytes())

    def region(self, source: Any, path: str, *, joined_effects: bool = False) -> RegionView:
        if type(source) is core.ClosedJaxpr:
            graph, constants = source.jaxpr, source.consts
        else:
            require(type(source) is core.Jaxpr, path, "expected Jaxpr/ClosedJaxpr")
            graph, constants = source, ()
        if len(graph.constvars) != len(constants):
            raise InvalidJaxprError(path, "unbound captured constants")
        variables: dict[core.Var, ValueDef] = {}

        def define(var: Any) -> ValueDef:
            if var in variables:
                raise InvalidJaxprError(path, "duplicate SSA definition")
            result = ValueDef(
                len(variables), self.value_type(var.aval, path), isinstance(var, core.DropVar)
            )
            variables[var] = result
            return result

        def operand(var: Any) -> Operand:
            if type(var) is core.Literal:
                return self.literal(var.val, var.aval, path)
            if var not in variables:
                raise InvalidJaxprError(path, "undefined SSA operand")
            return ValueRef(variables[var].id)

        def effects(items: Any, location: str) -> tuple[Effect, ...]:
            result: list[Effect] = []
            for effect in items:
                if type(effect) in (pp.SemaphoreEffect, pc.CommsEffect):
                    result.append(
                        GlobalEffectView(
                            "semaphore" if type(effect) is pp.SemaphoreEffect else "communication"
                        )
                    )
                    continue
                if type(effect) is core.NamedAxisEffect:
                    require(type(effect.name) is str, location, "unknown named-axis effect")
                    result.append(AxisEffectView(cast(str, effect.name)))
                    continue
                require(
                    type(effect) in (st.ReadEffect, st.WriteEffect), location, "unsupported effect"
                )
                ref = operand(effect.input)
                require(isinstance(ref, ValueRef), location, "effect must bind a Ref")
                require(
                    isinstance(variables[effect.input].type, RefType), location, "effect on non-Ref"
                )
                result.append(
                    EffectView(
                        "read" if type(effect) is st.ReadEffect else "write", cast(ValueRef, ref)
                    )
                )
            return tuple(
                sorted(
                    result,
                    key=lambda e: (
                        (0, e.ref.id, e.kind)
                        if isinstance(e, EffectView)
                        else (1, 0, e.name)
                        if isinstance(e, AxisEffectView)
                        else (2, 0, e.kind)
                    ),
                )
            )

        consts = tuple(
            (define(v), self.literal(c, v.aval, path))
            for v, c in zip(graph.constvars, constants, strict=True)
        )
        inputs = tuple(define(v) for v in graph.invars)
        operations = []
        for i, eq in enumerate(graph.eqns):
            loc = f"{path}.eqns[{i}]"
            require(
                attribute(eq.ctx, loc) == attribute(core.JaxprEqnContext(), loc),
                loc,
                "non-default equation context is not yet admitted",
            )
            args = tuple(operand(v) for v in eq.invars)
            eff = effects(eq.effects, loc)
            call, access, child, shard, dma, semaphore = None, None, None, None, None, None
            branches = ()
            loop_regions = ()
            allocations = ()
            if eq.primitive is pallas_call_p:
                self.calls += 1
                require(self.calls == 1, loc, "only one static pallas_call is supported")
                call = adapt_call(eq, self, loc)
                params = ()
                require(not eff, loc, "effectful outer pallas_call")
            elif eq.primitive is shard_map.shard_map_p:
                require(not eff, loc, "effectful shard_map call")
                shard, child = adapt_shard_map(eq, self, loc)
                params = ()
            elif eq.primitive in (
                custom_derivatives.custom_jvp_call_p,
                custom_derivatives.custom_vjp_call_p,
            ):
                require(not eff, loc, "effectful custom derivative forward wrapper")
                p = semantic_parameters(eq, include_derivative_rules=False)
                child = self.region(p["call_jaxpr"], loc + ".forward")
                require(
                    not child.effects
                    and not child.is_high
                    and tuple(v.type for v in child.inputs)
                    == tuple(self.value_type(v.aval, loc) for v in eq.invars),
                    loc,
                    "custom derivative forward binding differs",
                )
                params = tuple(
                    ParameterView(k, attribute(v, loc + "." + k))
                    for k, v in sorted(p.items())
                    if k != "call_jaxpr"
                )
            elif eq.primitive is pjit.jit_p:
                p = eq.params
                require(set(p) == JIT_SCHEMA, loc, "unaudited jit parameter schema")
                require(
                    (not eff or self.in_pallas_kernel)
                    and all(s is UNSPECIFIED for s in (*p["in_shardings"], *p["out_shardings"]))
                    and all(s is None for s in (*p["in_layouts"], *p["out_layouts"]))
                    and all(s is False for s in p["donated_invars"])
                    and p["ctx_mesh"].empty
                    and p["keep_unused"] is False
                    and p["inline"] is Inline.AUTO
                    and p["compiler_options_kvs"] == (),
                    loc,
                    "non-default JIT execution contract is not yet admitted",
                )
                child = self.region(p["jaxpr"], loc + ".call")
                require(not child.is_high, loc, "high-level JIT region")
                require(
                    tuple(v.type for v in child.inputs)
                    == tuple(self.value_type(v.aval, loc) for v in eq.invars),
                    loc,
                    "JIT argument binding types differ",
                )
                bindings = dict(zip((v.id for v in child.inputs), args, strict=True))
                rebound = set()
                for effect in child.effects:
                    if isinstance(effect, EffectView):
                        ref = bindings.get(effect.ref.id)
                        require(isinstance(ref, ValueRef), loc, "JIT effect on captured Ref")
                        rebound.add(EffectView(effect.kind, cast(ValueRef, ref)))
                    else:
                        rebound.add(effect)
                require(set(eff) == rebound, loc, "JIT effects do not bind to operands")
                params = tuple(
                    ParameterView(k, attribute(v, loc + "." + k))
                    for k, v in sorted(p.items())
                    if k != "jaxpr"
                )
            elif eq.primitive is pjit.reshard_p:
                params = adapt_local_reshard(eq, self, loc)
            elif eq.primitive in DMA_SCHEMAS:
                dma = adapt_dma(eq, args, eff, loc)
                params = ()
            elif eq.primitive is pp.run_scoped_p:
                child, allocations = adapt_scope(eq, self, args, eff, loc)
                params = ()
            elif eq.primitive in SEMAPHORE_SCHEMAS:
                semaphore = adapt_semaphore(eq, args, eff, loc)
                params = ()
            elif eq.primitive is parallel.axis_index_p:
                require(
                    set(eq.params) == {"axis_name"}
                    and type(eq.params["axis_name"]) is str
                    and not args
                    and eff == (AxisEffectView(eq.params["axis_name"]),),
                    loc,
                    "unaudited axis_index schema/effects",
                )
                try:
                    core.axis_frame(eq.params["axis_name"])
                except NameError as error:
                    require(False, loc, f"unbound device axis: {error}")
                params = (ParameterView("axis_name", eq.params["axis_name"]),)
            elif eq.primitive in (lax.scan_p, lax.while_p):
                params, loop_regions = adapt_loop(eq, self, args, eff, loc)
            elif eq.primitive is lax.cond_p:
                require(set(eq.params) == {"branches"}, loc, "unaudited conditional schema")
                sources = eq.params["branches"]
                require(type(sources) is tuple and bool(sources), loc, "invalid branch collection")
                branches = tuple(
                    self.region(branch, f"{loc}.branches[{j}]", joined_effects=True)
                    for j, branch in enumerate(sources)
                )
                branch_effects: set[Effect] = set()
                operation_effects: set[Effect] = set()
                for branch, source_branch in zip(branches, sources, strict=True):
                    require(
                        not branch.is_high
                        and tuple(v.type for v in branch.inputs)
                        == tuple(self.value_type(v.aval, loc) for v in eq.invars[1:])
                        and all(
                            left == right
                            or isinstance(left, ArrayType)
                            and isinstance(right, ArrayType)
                            and replace(left, weak_type=right.weak_type) == right
                            for left, right in zip(
                                (self.value_type(v.aval, loc) for v in source_branch.jaxpr.outvars),
                                (self.value_type(v.aval, loc) for v in eq.outvars),
                                strict=True,
                            )
                        ),
                        loc,
                        "conditional branch port types differ",
                    )
                    bindings = dict(zip((v.id for v in branch.inputs), args[1:], strict=True))
                    for effect in branch.effects:
                        if isinstance(effect, AxisEffectView | GlobalEffectView):
                            branch_effects.add(effect)
                        else:
                            ref = bindings.get(effect.ref.id)
                            require(isinstance(ref, ValueRef), loc, "branch effect on captured Ref")
                            branch_effects.add(EffectView(effect.kind, cast(ValueRef, ref)))
                    for operation in branch.operations:
                        for effect in operation.effects:
                            operation_effects.add(
                                EffectView(effect.kind, cast(ValueRef, bindings[effect.ref.id]))
                                if isinstance(effect, EffectView)
                                else effect
                            )
                require(
                    set(eff) == branch_effects == operation_effects,
                    loc,
                    "conditional effects do not bind to operands",
                )
                params = ()
            else:
                schema = {**PURE_SCHEMAS, **MEMORY_SCHEMAS, **COORDINATE_SCHEMAS}.get(eq.primitive)
                require(schema is not None, loc, f"unsupported primitive {eq.primitive.name}")
                require(set(eq.params) == schema, loc, "unaudited primitive parameter schema")
                if eq.primitive in MEMORY_SCHEMAS:
                    access = adapt_memory_access(eq, args, loc)
                    expected = "read" if eq.primitive in (sp.get_p, pp.load_p) else "write"
                    require(
                        eff == (EffectView(expected, access.ref),), loc, "inconsistent Ref effects"
                    )
                    params = tuple(
                        ParameterView(k, attribute(v, loc + "." + k))
                        for k, v in sorted(eq.params.items())
                        if k not in ("tree", "args_tree")
                    )
                else:
                    require(not eff, loc, "effectful pure operation")
                    explicit_broadcast = (
                        eq.primitive is lax.broadcast_in_dim_p
                        and eq.params.get("sharding") is not None
                    )
                    require(
                        eq.params.get("sharding") is None or explicit_broadcast,
                        loc,
                        "explicit primitive sharding is not yet admitted",
                    )
                    params = tuple(
                        ParameterView(k, local_broadcast_sharding(eq, self, loc))
                        if k == "sharding" and explicit_broadcast
                        else parameter(k, v, loc + "." + k)
                        for k, v in sorted(eq.params.items())
                    )
                    if explicit_broadcast:
                        params += (ParameterView("__pallas_kernel_broadcast", True),)
            results = tuple(define(v) for v in eq.outvars)
            regions = (
                (RegionBinding("call", child, args, tuple(ValueRef(v.id) for v in results)),)
                if child is not None
                else ()
            )
            if branches:
                regions = tuple(
                    RegionBinding(
                        f"branch[{j}]", branch, args[1:], tuple(ValueRef(v.id) for v in results)
                    )
                    for j, branch in enumerate(branches)
                )
            if loop_regions:
                regions = tuple(
                    RegionBinding(role, region, arguments, tuple(ValueRef(v.id) for v in results))
                    for role, region, arguments in loop_regions
                )
            operations.append(
                OperationView(
                    i,
                    eq.primitive.name,
                    args,
                    results,
                    params,
                    eff,
                    attribute(eq.ctx, loc + ".context"),
                    Origin(loc),
                    access,
                    call,
                    regions,
                    shard,
                    dma,
                    semaphore,
                    allocations,
                )
            )
        region_effects = effects(graph.effects, path)
        operation_effects = {e for op in operations for e in op.effects}
        require(
            operation_effects <= set(region_effects)
            if joined_effects
            else set(region_effects) == operation_effects,
            path,
            "inconsistent region effects",
        )
        return RegionView(
            inputs,
            consts,
            tuple(operations),
            tuple(operand(v) for v in graph.outvars),
            region_effects,
            graph.is_high,
            Origin(path),
        )


def adapt(source: Any, *, context: AdaptationContext | None = None) -> AdaptedProgram:
    """Adapt a supplied forward graph; no execution or tracing is performed."""
    context = context or AdaptationContext()
    require(
        (jax.__version__, jaxlib.__version__)
        == (context.jax_version, context.jaxlib_version)
        == ("0.11.2", "0.11.2"),
        "context",
        "unaudited JAX/jaxlib version",
    )
    require(context.derivative_policy == "forward", "context", "Pallas view supports forward only")
    adapter = JaxAdapter(context)
    graph = source.jaxpr if type(source) is core.ClosedJaxpr else source
    require(type(graph) is core.Jaxpr, "jaxpr", "expected Jaxpr/ClosedJaxpr")
    try:
        core.check_jaxpr(graph)
    except Exception as error:
        raise InvalidJaxprError("jaxpr", f"JAX type check failed: {error}") from error
    try:
        root = adapter.region(source, "jaxpr")
    except RecursionError as error:
        raise ComparisonBudgetError(
            "jaxpr",
            "source adaptation nesting limit exceeded",
            failure_kind=ComparisonFailureKind.RECURSION_LIMIT,
        ) from error
    return AdaptedProgram(ProgramView(root, context), SourceHandle(source))
