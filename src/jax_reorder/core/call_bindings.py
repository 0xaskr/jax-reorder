"""Extract a single Pallas call with its lexical value and named-axis bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, axis
from .memory import Expression, ValueExpr, _literal, _substitute, scalar_type
from .model import (
    ArrayType,
    Attribute,
    LiteralValue,
    Operand,
    OperationView,
    Origin,
    ParameterView,
    ProgramView,
    RegionView,
    ValueDef,
    ValueRef,
    ValueType,
)
from .tensor_domain import coordinate, coordinate_ranges, requires_tensor_coordinates
from .tensor_values import tensor_value
from .value_operations import pure_value


@dataclass(frozen=True, eq=False)
class BindingExpr:
    operation: str
    type: ValueType
    operands: tuple[BindingExpr, ...] = ()
    parameters: tuple[ParameterView, ...] = ()
    data: bytes = b""
    port: int | None = None
    context: Attribute = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BindingExpr):
            return NotImplemented
        pending: list[tuple[BindingExpr, BindingExpr]] = [(self, other)]
        seen: set[tuple[int, int]] = set()
        while pending:
            a, b = pending.pop()
            pair = id(a), id(b)
            if pair in seen:
                continue
            seen.add(pair)
            if (a.operation, a.type, a.parameters, a.data, a.port, a.context) != (
                b.operation,
                b.type,
                b.parameters,
                b.data,
                b.port,
                b.context,
            ) or len(a.operands) != len(b.operands):
                return False
            pending.extend(zip(a.operands, b.operands, strict=True))
        return True

    def __hash__(self) -> int:
        pending: list[tuple[BindingExpr, bool]] = [(self, False)]
        done: dict[int, int] = {}
        while pending:
            value, ready = pending.pop()
            if id(value) in done:
                continue
            if not ready:
                pending.append((value, True))
                pending.extend((v, False) for v in value.operands)
            else:
                done[id(value)] = hash(
                    (
                        value.operation,
                        value.type,
                        value.parameters,
                        value.data,
                        value.port,
                        value.context,
                        tuple(done[id(v)] for v in value.operands),
                    )
                )
        return done[id(self)]


@dataclass(frozen=True)
class CallSite:
    operation: OperationView
    arguments: tuple[BindingExpr, ...]
    wrappers: tuple[OperationView, ...]
    outputs: tuple[BindingExpr, ...]


def extract_call(view: ProgramView, *, budget: AnalysisBudget) -> CallSite:
    found: list[tuple[OperationView, tuple[BindingExpr, ...], tuple[OperationView, ...]]] = []

    def region(
        source: RegionView, args: tuple[BindingExpr, ...], wrappers: tuple[OperationView, ...]
    ) -> tuple[BindingExpr, ...]:
        env = dict(zip((v.id for v in source.inputs), args, strict=True))

        def literal(v: LiteralValue) -> BindingExpr:
            return BindingExpr("literal", v.type, data=v.data)

        env.update((v.id, literal(c)) for v, c in source.constants)

        def read(v: Operand) -> BindingExpr:
            return env[v.id] if isinstance(v, ValueRef) else literal(v)

        for op in source.operations:
            budget.spend(op.origin.path)
            values = tuple(read(v) for v in op.operands)
            if op.call is not None:
                found.append((op, values, wrappers))
                outputs = tuple(
                    BindingExpr("kernel_output", v.type, port=i) for i, v in enumerate(op.results)
                )
            elif op.regions:
                if (
                    op.primitive not in ("jit", "shard_map", "custom_jvp_call", "custom_vjp_call")
                    or len(op.regions) != 1
                ):
                    raise UnsupportedJaxprError(
                        op.origin.path, "call extraction requires reviewed static wrappers"
                    )
                child = op.regions[0]
                if op.shard_map is not None:
                    # The local interface is the shard-local value. Keep the
                    # original provenance underneath for replay and diagnostics.
                    values = tuple(
                        BindingExpr(
                            "local_input",
                            v.type,
                            (a,),
                            (ParameterView("input_partition", partition),),
                        )
                        for v, a, partition in zip(
                            child.region.inputs, values, op.shard_map.input_partitions, strict=True
                        )
                    )
                outputs = region(child.region, values, wrappers + (op,))
                if op.shard_map is not None:
                    outputs = tuple(
                        replace(a, type=v.type) for a, v in zip(outputs, op.results, strict=True)
                    )
            else:
                if op.access or op.dma or op.semaphore or len(op.results) != 1:
                    raise UnsupportedJaxprError(
                        op.origin.path, "unsupported outer value/effect binding"
                    )
                outputs = (
                    BindingExpr(
                        op.primitive, op.results[0].type, values, op.parameters, context=op.context
                    ),
                )
            env.update((v.id, a) for v, a in zip(op.results, outputs, strict=True))
        return tuple(read(o) for o in source.outputs)

    outputs = region(
        view.root,
        tuple(BindingExpr("input", v.type, port=i) for i, v in enumerate(view.root.inputs)),
        (),
    )
    if len(found) != 1:
        raise UnsupportedJaxprError(view.root.origin.path, "expected one static pallas_call")
    op, arguments, wrappers = found[0]
    return CallSite(op, arguments, wrappers, outputs)


def binding_value(
    binding: BindingExpr,
    *,
    budget: AnalysisBudget,
    coordinates: dict[str, IndexExpr] | None = None,
    ranges: dict[str, tuple[int, int]] | None = None,
) -> Expression:
    cache: dict[int, Expression] = {}
    pending, seen, tensor = [binding], set(), False
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        budget.spend("call.binding_shapes")
        tensor |= requires_tensor_coordinates(
            item.operation, tuple(a.type for a in item.operands), (item.type,)
        )
        pending.extend(item.operands)

    def emit(v: BindingExpr) -> Expression:
        if id(v) in cache:
            return cache[id(v)]
        budget.spend("call.bindings")
        if not isinstance(v.type, ArrayType):
            raise UnsupportedJaxprError("call.bindings", "outer value requires array type")
        typ = v.type
        addresses = (
            tuple(coordinate("local", i) for i in range(len(typ.shape)))
            if tensor
            else ((axis("global"),) if typ.shape else ())
        )
        if v.operation == "input":
            result = ValueExpr("input", scalar_type(typ), addresses, input=v.port)
        elif v.operation == "literal":
            result = (
                _literal(LiteralValue(typ, v.data), "call.bindings")
                if not typ.shape
                else ValueExpr(
                    "constant_array",
                    scalar_type(typ),
                    addresses,
                    (ParameterView("shape", typ.shape),),
                    v.data,
                )
            )
        elif v.operation == "local_input":
            raise UnsupportedJaxprError(
                "call.bindings", "outer shard binding needs global assembly or local interface"
            )
        elif v.operation == "kernel_output":
            raise UnsupportedJaxprError(
                "call.bindings", "outer postprocessing needs output composition"
            )
        else:
            op = OperationView(
                0,
                v.operation,
                (),
                (ValueDef(0, typ),),
                v.parameters,
                (),
                v.context,
                Origin("call.bindings"),
            )
            if tensor:
                input_types = tuple(a.type for a in v.operands if isinstance(a.type, ArrayType))
                if len(input_types) != len(v.operands):
                    raise UnsupportedJaxprError("call.bindings", "non-array parent operand")
                result = tensor_value(
                    op,
                    tuple(emit(a) for a in v.operands),
                    input_types,
                    {**(ranges or {}), **coordinate_ranges("local", typ.shape)},
                    coordinates or {},
                    (),
                    budget,
                )
            else:
                value_ranges = dict(ranges or {})
                if typ.shape:
                    # Bound this intermediate's lane, not the eventual call shape.
                    value_ranges["global"] = (0, typ.shape[0] - 1)
                result = pure_value(
                    op, tuple(emit(a) for a in v.operands), value_ranges, coordinates or {}
                )
        cache[id(v)] = result
        return result

    result = emit(binding)
    if tensor and isinstance(binding.type, ArrayType) and len(binding.type.shape) == 1:
        return _substitute(result, {"local:0": axis("global")})
    return result
