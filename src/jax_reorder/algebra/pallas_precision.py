"""Conservative pairing of non-default dot policies before real interpretation."""

from typing import Any, cast

import numpy as np

from ..core.index_domain import IndexExpr
from ..core.memory import Expression, MemoryProjection, Predicate, ValueExpr
from ..errors import UnsupportedJaxprError
from ._typing import SymbolicGraph
from .budget import Budget

_HIGHEST = (
    ("type", "builtins", "tuple"),
    (
        ("enum", ("type", "jax._src.lax.lax", "Precision"), (("type", "builtins", "int"), 2)),
        ("enum", ("type", "jax._src.lax.lax", "Precision"), (("type", "builtins", "int"), 2)),
    ),
)


def highest_precision(value: object) -> bool:
    return value == _HIGHEST


def pair_pallas_precision(left: SymbolicGraph, right: SymbolicGraph, budget: Budget) -> bool:
    """Pair complete dot identities, allowing only scalar-factor relocation.

    This is an admission condition, not an equality proof. Real coefficients
    stay in the actual formulas. Other changes to a mixed-policy contraction's
    operands conservatively need a new correspondence proof.
    """
    projections = [
        tuple(
            eq.params["projection"]
            for eq in graph.eqns
            if eq.primitive.name == "pallas_global_values"
        )
        for graph in (left, right)
    ]
    if not any(projections):
        return False
    identities: dict[tuple[Any, ...], int] = {}
    nondefault = False

    def inventory(projection: MemoryProjection) -> tuple[frozenset[int], tuple[int, ...]]:
        nonlocal nondefault
        cache: dict[int, int] = {}
        dots: set[int] = set()

        def visit(value: Expression) -> int:
            nonlocal nondefault
            if id(value) in cache:
                return cache[id(value)]
            budget.spend(path="algebra.pallas.precision_pairing")
            if isinstance(value, ValueExpr):
                children = tuple(visit(child) for child in value.operands)
                if value.operation == "mul":
                    # Scalar floating coefficients may move into/out of dot in
                    # the real model. Never erase a cast, array or data operand.
                    for i, operand in enumerate(value.operands):
                        if (
                            isinstance(operand, ValueExpr)
                            and operand.operation == "literal"
                            and operand.type == value.type
                            and operand.type.dtype in ("float16", "float32", "float64")
                            and np.isfinite(
                                np.frombuffer(operand.data, dtype=operand.type.dtype)
                            ).all()
                            and len(children) == 2
                        ):
                            cache[id(value)] = children[1 - i]
                            return children[1 - i]
                key = (
                    value.operation,
                    value.type,
                    value.parameters,
                    value.data,
                    value.input,
                    children,
                )
                if value.operation == "tensor.dot_general":
                    params = {p.name: p.value for p in value.parameters}
                    precision = params.get("precision")
                    if precision != ("none",):
                        nondefault = True
                        if not highest_precision(precision) or any(
                            dtype not in ("float16", "bfloat16", "float32", "float64")
                            for dtype in (
                                *cast(tuple[str, ...], params["__input_dtypes"]),
                                value.type.dtype,
                            )
                        ):
                            raise UnsupportedJaxprError(
                                "algebra.pallas.precision", "unreviewed non-default dot precision"
                            )
                    dots.add(identities.setdefault(key, len(identities)))
            elif isinstance(value, Predicate):
                key = ("predicate", value.operation, tuple(visit(v) for v in value.operands))
            elif isinstance(value, IndexExpr):
                key = ("index", value.kind, value.value, tuple(visit(v) for v in value.args))
            else:
                key = ("bool", value)
            identity = identities.setdefault(key, len(identities))
            cache[id(value)] = identity
            return identity

        # Ordered roots retain where each policy is observed; a set alone could
        # hide DEFAULT/HIGHEST being exchanged between two output ports.
        roots = tuple(visit(root) for root in projection.outputs)
        return frozenset(dots), roots

    paired = [tuple(inventory(projection) for projection in group) for group in projections]
    if nondefault and paired[0] != paired[1]:
        raise UnsupportedJaxprError(
            "algebra.pallas.precision",
            "non-default dot policies lack matching operand/type correspondence",
        )
    return nondefault
