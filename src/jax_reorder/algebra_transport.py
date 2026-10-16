"""Freeze already-validated jaxprs for an isolated symbolic worker, without JAX execution."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from jax._src import core

from .derivatives import CUSTOM_PRIMITIVES, semantic_parameters
from .errors import UnsupportedJaxprError


@dataclass(frozen=True)
class Variable:
    number: int
    shape: tuple
    dtype: object

    @property
    def aval(self):
        return SimpleNamespace(shape=self.shape, dtype=self.dtype)


def freeze(graph, common_calls=None, common_ids=None):
    """Only the mathematical builder's supported parameters cross the process boundary."""
    closed = hasattr(graph, "jaxpr")
    if common_ids is None:
        common_ids = {}
    inner = graph.jaxpr if closed else graph
    if inner.effects:
        raise UnsupportedJaxprError("algebra.effects", "effectful graph")
    variables = {}

    def variable(var):
        if not isinstance(var.aval, core.ShapedArray):
            raise UnsupportedJaxprError(
                "algebra.shape", "requires shaped array values; tokens are unsupported"
            )
        if isinstance(var, core.Literal):
            return SimpleNamespace(val=np.asarray(var.val), aval=var.aval)
        if var not in variables:
            variables[var] = Variable(len(variables), tuple(var.aval.shape), var.aval.dtype)
        return variables[var]

    def parameter(value):
        if isinstance(value, (core.Jaxpr, core.ClosedJaxpr)):
            return freeze(value, common_calls, common_ids)
        if isinstance(value, (tuple, list)):
            return tuple(parameter(x) for x in value)
        if isinstance(value, dict):
            return {key: parameter(x) for key, x in value.items()}
        return value

    inputs = tuple(variable(v) for v in inner.invars)
    constants = tuple(variable(v) for v in inner.constvars)
    equations = []
    for eq in inner.eqns:
        if eq.effects:
            raise UnsupportedJaxprError("algebra.effects", "effectful equation")
        # JIT layout/name/compiler metadata has already been validated structurally.
        # It is not part of the declared ideal-arithmetic interpretation.
        params = semantic_parameters(eq, scope="primal", parent=graph)
        if eq.primitive in CUSTOM_PRIMITIVES:
            params = {"call_jaxpr": params["call_jaxpr"]}
        if eq.primitive.name == "reduce":
            from .canonical import canonicalize
            from .correspondence import _parameter_view

            encoded, children = _parameter_view(
                eq, lambda _: None, derivative_scope="primal", parent=graph
            )
            key = (
                "opaque_reducer",
                encoded,
                tuple(
                    (path, canonicalize(child, derivative_scope="primal").graph)
                    for path, child in children.items()
                ),
            )
            if key not in common_ids:
                common_ids[key] = len(common_ids)
            params = {"opaque_semantic": common_ids[key]}
        if eq.primitive.name == "convert_element_type" and params.get("sharding") is not None:
            from .algebra_lower import encode

            aval = eq.outvars[0].aval
            default = core.ShapedArray(aval.shape, aval.dtype).sharding
            if encode(params["sharding"]) != encode(default):
                raise UnsupportedJaxprError("algebra.cast", "non-default cast sharding")
            params = {**params, "sharding": None}
        if eq.primitive.name == "scan":
            ft_in, ft_out = params["ft_in"], params["ft_out"]
            if (
                not ft_in.unpackable
                or not ft_out.unpackable
                or any(value is not None for value in (*ft_in.vals, *ft_out.vals))
            ):
                raise UnsupportedJaxprError("algebra.scan", "unsupported scan interface metadata")
            scan_constants, carries, _ = ft_in.unpack()
            params = {key: value for key, value in params.items() if key not in ("ft_in", "ft_out")}
            params.update(num_consts=len(scan_constants), num_carry=len(carries))
        if eq.primitive.name in ("jit", "pjit"):
            from .algebra_jit import default_jit

            if not default_jit(eq, params):
                raise UnsupportedJaxprError("algebra.jit", "non-default JIT execution contract")
            params = {"jaxpr": params["jaxpr"]}
            if common_calls is not None and eq in common_calls.invocations:
                from .algebra_lower import Lowerer

                key, input_ports, output_ports = common_calls.invocations[eq]
                if not Lowerer(common_calls.budget)._nonfinite_graph(params["jaxpr"]):
                    if key not in common_ids:
                        common_ids[key] = len(common_ids)
                    params = {**params, "common_call": (common_ids[key], input_ports, output_ports)}
        equations.append(
            SimpleNamespace(
                primitive=SimpleNamespace(name=eq.primitive.name),
                params=parameter(params),
                invars=tuple(variable(v) for v in eq.invars),
                outvars=tuple(variable(v) for v in eq.outvars),
            )
        )
    return SimpleNamespace(
        invars=inputs,
        constvars=constants,
        consts=tuple(np.asarray(x) for x in graph.consts) if closed else (),
        eqns=tuple(equations),
        outvars=tuple(variable(v) for v in inner.outvars),
    )
