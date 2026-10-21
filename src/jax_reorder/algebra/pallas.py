"""Build complete output formulas from source-free, proven memory projections.

This module constructs formulas only. Equality and definedness decisions remain
in trusted.py, including the requirement for a complete-output counterexample.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import sympy as sp

from ..core.index_domain import IndexExpr
from ..core.memory import Expression, MemoryProjection, Predicate, ValueExpr
from ..errors import UnsupportedJaxprError
from ._typing import ArrayAval, Equation
from .expression import SymbolicTensor, coordinates, rename_bounds
from .finiteness import Source
from .pallas_domains import lower_padding_domains

if TYPE_CHECKING:
    from .symbolic import SymbolicBuilder


def _memory_read(value: SymbolicTensor, indices: Sequence[sp.Expr | int]) -> sp.Basic:
    replacements = dict(zip(coordinates(value.shape), map(sp.sympify, indices), strict=True))
    replacements.pop(sp.S.Zero, None)
    return rename_bounds(value.memory_validity, fresh=True).xreplace(replacements)


def _memory_broadcast(value: SymbolicTensor, shape: tuple[int, ...]) -> sp.Basic:
    indices = coordinates(shape)[len(shape) - len(value.shape) :]
    return _memory_read(
        value, tuple(0 if n == 1 else i for n, i in zip(value.shape, indices, strict=True))
    )


def _operation(
    builder: SymbolicBuilder, equation: Equation, values: Sequence[SymbolicTensor]
) -> SymbolicTensor:
    """Keep initialization independent of partial real-function domains.

    Core already scalarizes views and broadcasts. The only remaining cross-lane
    value operations are reviewed reductions and dot; their original full bounds
    are used for both value formulas and the universal initialization obligation.
    """
    (result,) = builder.operation(
        equation, [replace(v, memory_validity=sp.true) for v in values], 0
    )
    if all(v.memory_validity == sp.true for v in values):
        return result
    shape, name = result.shape, equation.primitive.name
    axes, params = coordinates(shape), equation.params
    if name == "select_n":
        predicate = builder.broadcast(values[0], shape)[0]
        boolean = np.dtype(values[0].dtype).kind == "b"
        guards = [
            (sp.Not(predicate) if i == 0 else predicate) if boolean else sp.Eq(predicate, i)
            for i in range(len(values) - 1)
        ]
        valid = sp.And(
            _memory_broadcast(values[0], shape),
            sp.Or(
                *(
                    sp.And(guard, _memory_broadcast(value, shape))
                    for guard, value in zip(guards, values[1:], strict=True)
                )
            ),
        )
    elif name in ("reduce_sum", "reduce_max"):
        original = values[0]
        limits = [builder.bound(original.shape[axis]) for axis in params["axes"]]
        reductions = dict(zip(params["axes"], limits, strict=True))
        free = iter(axes)
        indices = tuple(
            reductions[i][0] if i in reductions else next(free) for i in range(len(original.shape))
        )
        valid = builder.universal(_memory_read(original, indices), limits)
    elif name == "dot_general":
        left, right = values
        (ca, cb), (ba, bb) = params["dimension_numbers"]
        ia, ib = [sp.S.Zero] * len(left.shape), [sp.S.Zero] * len(right.shape)
        free = iter(axes)
        for a, b in zip(ba, bb, strict=True):
            ia[a] = ib[b] = next(free)
        for indices, contracted, batched in ((ia, ca, ba), (ib, cb, bb)):
            for axis in range(len(indices)):
                if axis not in contracted + batched:
                    indices[axis] = next(free)
        limits = []
        for a, b in zip(ca, cb, strict=True):
            limit = builder.bound(left.shape[a])
            ia[a] = ib[b] = limit[0]
            limits.append(limit)
        valid = builder.universal(sp.And(_memory_read(left, ia), _memory_read(right, ib)), limits)
    elif name in builder.pointwise_primitives or name in (
        "convert_element_type",
        "copy",
        "stop_gradient",
    ):
        valid = sp.And(*(_memory_broadcast(value, shape) for value in values))
    else:
        raise UnsupportedJaxprError(
            "algebra.pallas.initialization", f"no initialization rule for {name}"
        )
    return replace(result, memory_validity=valid)


def projected_outputs(
    builder: SymbolicBuilder, equation: Equation, inputs: Sequence[SymbolicTensor]
) -> list[SymbolicTensor]:
    projection: MemoryProjection = equation.params["projection"]
    builder.rules.add(equation.params["rule"])
    results = []
    for root, var in zip(projection.outputs, equation.outvars, strict=True):
        shape = tuple(var.aval.shape)
        has_padding = False
        # External output axes stay distinct from each tensor operation's local
        # axes. Otherwise a contraction's read() could capture an output axis.
        output_coordinates = tuple(
            sp.Dummy("output", integer=True, nonnegative=True) for _ in shape
        )
        bindings = {f"output:{i}": v for i, v in enumerate(output_coordinates)}
        if len(shape) == 1:
            bindings["global"] = output_coordinates[0]

        def scalar(value: Any, dtype: Any) -> SymbolicTensor:
            return builder.constant(value, cast(ArrayAval, SimpleNamespace(shape=(), dtype=dtype)))

        def emitter(
            shape: tuple[int, ...], bindings: dict[str, sp.Expr]
        ) -> Callable[[Expression], SymbolicTensor]:
            cache: dict[int, SymbolicTensor] = {}

            def emit(value: Expression) -> SymbolicTensor:
                nonlocal has_padding
                if id(value) in cache:
                    return cache[id(value)]
                builder.budget.spend(path="algebra.pallas.value")
                if type(value) is bool:
                    result = scalar(value, np.dtype("bool"))
                elif isinstance(value, IndexExpr):
                    children = [emit(v).expression for v in value.args]
                    if value.kind == "constant":
                        expression = sp.Integer(value.value)
                    elif value.kind == "axis" and str(value.value) in bindings:
                        expression = bindings[str(value.value)]
                    elif value.kind == "sum":
                        expression = sp.Add(*children)
                    elif value.kind == "scale":
                        expression = int(value.value) * children[0]
                    elif value.kind == "div":
                        expression = sp.floor(children[0] / int(value.value))
                    elif value.kind == "mod":
                        expression = sp.Mod(children[0], int(value.value))
                    elif value.kind == "lookup":
                        expression = sp.Piecewise(
                            *((v, sp.Eq(children[0], i)) for i, v in enumerate(children[1:-1])),
                            (children[-1], True),
                        )
                    else:
                        raise UnsupportedJaxprError(
                            "algebra.pallas.index", "unprojected coordinate"
                        )
                    result = SymbolicTensor(
                        shape,
                        np.dtype("int32"),
                        expression,
                        source=Source(None, shape, np.dtype("int32"), finite=True),
                    )
                elif isinstance(value, Predicate):
                    predicates = {
                        "lt": sp.Lt,
                        "le": sp.Le,
                        "gt": sp.Gt,
                        "ge": sp.Ge,
                        "eq": sp.Eq,
                        "ne": sp.Ne,
                        "and": sp.And,
                        "or": sp.Or,
                        "not": sp.Not,
                    }
                    if value.operation not in predicates:
                        raise UnsupportedJaxprError("algebra.pallas.predicate", "unknown predicate")
                    args = [emit(v) for v in value.operands]
                    result = SymbolicTensor(
                        shape,
                        np.dtype("bool"),
                        predicates[value.operation](*(v.expression for v in args)),
                        sp.And(*(v.defined for v in args)),
                        memory_validity=sp.And(*(v.memory_validity for v in args)),
                    )
                elif isinstance(value, ValueExpr):
                    dtype = np.dtype(value.type.dtype)
                    if value.operation == "padded_value":
                        has_padding = True
                        item, valid = (emit(v) for v in value.operands)
                        result = replace(
                            item,
                            memory_validity=sp.And(item.memory_validity, valid.expression),
                        )
                        builder.rules.add("pallas.input-padding-validity:v1")
                    elif value.operation == "literal":
                        result = scalar(np.frombuffer(value.data, dtype=dtype).reshape(()), dtype)
                    elif value.operation in ("input", "constant_array"):
                        if value.operation == "input" and value.input is None:
                            raise UnsupportedJaxprError(
                                "algebra.pallas.input", "invalid input binding"
                            )
                        if value.operation == "input":
                            assert value.input is not None
                            source = inputs[value.input]
                        else:
                            constant_shape = dict((p.name, p.value) for p in value.parameters)[
                                "shape"
                            ]
                            array = np.frombuffer(value.data, dtype=dtype).reshape(
                                cast(tuple[int, ...], constant_shape)
                            )
                            source = builder.constant(
                                array,
                                cast(ArrayAval, SimpleNamespace(shape=array.shape, dtype=dtype)),
                            )
                        addresses = tuple(emit(a).expression for a in value.operands)
                        expr, defined = source.read(addresses)
                        # Different global reads must not share a Source identity
                        # merely because they read the same external array slot.
                        result = replace(
                            source,
                            shape=shape,
                            expression=expr,
                            defined=defined,
                            memory_validity=_memory_read(source, addresses),
                            source=Source(None, shape, source.dtype, finite=source.finite),
                        )
                    elif (
                        value.operation == "select_n"
                        and dtype.kind in "biu"
                        and len(value.operands) == 3
                        and isinstance(value.operands[0], Predicate | bool)
                    ):
                        # Core memory assembly selects an already proved piece
                        # using a geometric/Boolean predicate. It performs no
                        # integer arithmetic, so the generic opaque integer
                        # primitive policy must not hide this exact selection.
                        args = [emit(v) for v in value.operands]
                        (pred, dp), (left, dl), (right, dr) = (
                            builder.broadcast(v, shape) for v in args
                        )
                        result = SymbolicTensor(
                            shape,
                            dtype,
                            sp.Piecewise((right, pred), (left, True)),
                            sp.And(
                                dp,
                                dl
                                if dl == dr
                                else sp.Or(sp.And(pred, dr), sp.And(sp.Not(pred), dl)),
                            ),
                            finite=all(v.finite for v in args[1:]),
                            memory_validity=sp.And(
                                _memory_broadcast(args[0], shape),
                                sp.Or(
                                    sp.And(pred, _memory_broadcast(args[2], shape)),
                                    sp.And(sp.Not(pred), _memory_broadcast(args[1], shape)),
                                ),
                            ),
                        )
                        builder.rules.add("pallas.integer-memory-selection:v1")
                    elif value.operation.startswith("tensor."):
                        metadata = {
                            p.name: p.value for p in value.parameters if p.name.startswith("__")
                        }
                        input_shapes = cast(tuple[tuple[int, ...], ...], metadata["__input_shapes"])
                        input_dtypes = cast(tuple[str, ...], metadata["__input_dtypes"])
                        input_axes = cast(tuple[tuple[str, ...], ...], metadata["__input_axes"])
                        output_shape = cast(tuple[int, ...], metadata["__output_shape"])
                        args = []
                        for body, input_shape, input_dtype, names in zip(
                            value.operands[: len(input_shapes)],
                            input_shapes,
                            input_dtypes,
                            input_axes,
                            strict=True,
                        ):
                            axes = tuple(
                                sp.Dummy("tensor", integer=True, nonnegative=True) for _ in names
                            )
                            local = emitter(
                                input_shape, {**bindings, **dict(zip(names, axes, strict=True))}
                            )(body)
                            rename = dict(zip(axes, coordinates(input_shape), strict=True))
                            args.append(
                                replace(
                                    local,
                                    shape=input_shape,
                                    dtype=np.dtype(input_dtype),
                                    expression=local.expression.xreplace(rename),
                                    defined=local.defined.xreplace(rename),
                                    memory_validity=local.memory_validity.xreplace(rename),
                                )
                            )
                        params = {}
                        for p in value.parameters:
                            if p.name.startswith("__"):
                                continue
                            if p.name in {"precision", "out_sharding"}:
                                if p.value != ("none",):
                                    raise UnsupportedJaxprError(
                                        "algebra.pallas.tensor", f"non-default {p.name}"
                                    )
                                params[p.name] = None
                            elif p.name == "preferred_element_type":
                                params[p.name] = (
                                    None if p.value == ("none",) else np.dtype(cast(str, p.value))
                                )
                            elif p.name in {"dimension_numbers", "axes"}:
                                params[p.name] = p.value
                            else:
                                raise UnsupportedJaxprError("algebra.pallas.tensor", p.name)
                        eq = cast(
                            Equation,
                            SimpleNamespace(
                                primitive=SimpleNamespace(
                                    name=value.operation.removeprefix("tensor.")
                                ),
                                params=params,
                                invars=(),
                                outvars=(
                                    SimpleNamespace(
                                        aval=SimpleNamespace(shape=output_shape, dtype=dtype)
                                    ),
                                ),
                            ),
                        )
                        computed = _operation(builder, eq, args)
                        at = tuple(emit(a).expression for a in value.operands[len(input_shapes) :])
                        expression, defined = computed.read(at)
                        result = replace(
                            computed,
                            shape=shape,
                            expression=expression,
                            defined=defined,
                            memory_validity=_memory_read(computed, at),
                            source=Source(None, shape, dtype, finite=computed.finite),
                        )
                    else:
                        args = [emit(v) for v in value.operands]
                        params = {}
                        for p in value.parameters:
                            if p.name in {"accuracy", "out_dtype", "sharding"}:
                                if p.value != ("none",):
                                    raise UnsupportedJaxprError(
                                        "algebra.pallas.parameter", f"non-default {p.name}"
                                    )
                                params[p.name] = None
                            elif p.name in {"y", "weak_type"}:
                                params[p.name] = p.value
                            elif p.name == "new_dtype":
                                params[p.name] = np.dtype(cast(str, p.value))
                            else:
                                raise UnsupportedJaxprError("algebra.pallas.parameter", p.name)
                        eq = cast(
                            Equation,
                            SimpleNamespace(
                                primitive=SimpleNamespace(name=value.operation),
                                params=params,
                                invars=(),
                                outvars=(
                                    SimpleNamespace(aval=SimpleNamespace(shape=shape, dtype=dtype)),
                                ),
                            ),
                        )
                        result = _operation(builder, eq, args)
                else:
                    raise UnsupportedJaxprError("algebra.pallas", "unknown projected value")
                cache[id(value)] = result
                return result

            return emit

        result = emitter(shape, bindings)(root)
        expr, defined = builder.broadcast(result, shape)
        rename = dict(zip(output_coordinates, coordinates(shape), strict=True))
        expr, defined = expr.xreplace(rename), defined.xreplace(rename)
        validity = _memory_broadcast(result, shape).xreplace(rename)
        # Tensor reads substitute local output coordinates after the core has
        # proved their global address domain. Compose the resulting integer
        # quotient/remainder spelling before comparing contraction formulas.
        reads = expr.atoms(sp.Indexed) | defined.atoms(sp.Indexed) | validity.atoms(sp.Indexed)
        builder.budget.spend(len(reads), path="algebra.pallas.address_spelling")
        replacements = {
            read: read.base[tuple(sp.expand(i.rewrite(sp.floor)) for i in read.indices)]
            for read in reads
        }
        expr, defined = expr.xreplace(replacements), defined.xreplace(replacements)
        validity = validity.xreplace(replacements)
        # IndexExpr intentionally models checked mathematical coordinates; the
        # enclosing output contract supplies the final integer dtype.
        result = replace(
            result,
            shape=shape,
            dtype=var.aval.dtype,
            expression=expr,
            defined=defined,
            memory_validity=validity,
        )
        results.append(lower_padding_domains(builder, result) if has_padding else result)
    return results
