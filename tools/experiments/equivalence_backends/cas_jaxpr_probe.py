"""Translate actual GLM helper jaxprs to indexed SymPy expressions, without execution.

Run as a module using the project interpreter, with the isolated SymPy site-packages
on PYTHONPATH. This deliberately small experimental adapter is not production code.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from unittest.mock import patch

import jax
import jax.numpy as jnp
import sympy as sp
from jax._src import compiler, core, dispatch

from examples import glm52_decode_dsa_mla as layer


class Unsupported(ValueError):
    pass


def forbidden(*args, **kwargs):
    raise AssertionError("JAX execution and numerical equality heuristics are forbidden")


@dataclass
class Lowering:
    graph: object

    def __post_init__(self):
        if self.graph.consts or self.graph.jaxpr.effects:
            raise Unsupported("captured constants and effects are outside this adapter")
        self.inputs = {var: index for index, var in enumerate(self.graph.jaxpr.invars)}
        self.equations = {}
        supported = {"add", "mul", "broadcast_in_dim", "dot_general", "reduce_sum"}
        supported.add("convert_element_type")
        for equation in self.graph.jaxpr.eqns:
            if equation.primitive.name not in supported or len(equation.outvars) != 1:
                raise Unsupported(f"unsupported primitive: {equation.primitive.name}")
            if equation.effects:
                raise Unsupported("effectful equation")
            self.equations[equation.outvars[0]] = equation
        for var in (*self.inputs, *self.equations):
            if any(not isinstance(size, int) or size <= 0 for size in var.aval.shape):
                raise Unsupported("only positive static dimensions; empty domains need a model")
            if str(var.aval.dtype) != "float32":
                raise Unsupported(f"only float32 interpreted as finite real: {var.aval.dtype}")

    def read(self, var, indices):
        if isinstance(var, core.Literal):
            if var.aval.shape or not math.isfinite(float(var.val)):
                raise Unsupported("only finite scalar literals")
            return sp.Rational(float(var.val))
        if len(indices) != len(var.aval.shape):
            raise Unsupported("index rank mismatch")
        if var in self.inputs:
            name = f"input{self.inputs[var]}"
            if not indices:
                return sp.Symbol(name, real=True)
            return sp.IndexedBase(name, shape=var.aval.shape, real=True)[indices]
        equation = self.equations[var]
        op, args = equation.primitive.name, equation.invars
        if op in {"add", "mul"}:
            parts = []
            for arg in args:
                shape = arg.aval.shape
                aligned = indices[len(indices) - len(shape) :] if shape else ()
                mapped = tuple(
                    0 if size == 1 else ix for ix, size in zip(aligned, shape, strict=True)
                )
                parts.append(self.read(arg, mapped))
            return sp.Add(*parts) if op == "add" else sp.Mul(*parts)
        if op == "broadcast_in_dim":
            dimensions = equation.params["broadcast_dimensions"]
            mapped = tuple(
                0 if size == 1 else indices[axis]
                for size, axis in zip(args[0].aval.shape, dimensions, strict=True)
            )
            return self.read(args[0], mapped)
        if op == "convert_element_type":
            if args[0].aval.dtype != var.aval.dtype:
                raise Unsupported("dtype-changing conversion needs an explicit mathematical model")
            return self.read(args[0], indices)
        if op == "reduce_sum":
            axes = tuple(equation.params["axes"])
            source_shape = args[0].aval.shape
            mapped, limits, free = [], [], iter(indices)
            for axis, size in enumerate(source_shape):
                if axis in axes:
                    bound = sp.Dummy("reduce", integer=True)
                    mapped.append(bound)
                    limits.append((bound, 0, size - 1))
                else:
                    mapped.append(next(free))
            return sp.Sum(self.read(args[0], tuple(mapped)), *limits)
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = equation.params["dimension_numbers"]
        lhs, rhs = args
        lhs_indices = [None] * len(lhs.aval.shape)
        rhs_indices = [None] * len(rhs.aval.shape)
        free, limits = iter(indices), []
        for la, ra in zip(lhs_batch, rhs_batch, strict=True):
            lhs_indices[la] = rhs_indices[ra] = next(free)
        for mapping, contract, batch in (
            (lhs_indices, lhs_contract, lhs_batch),
            (rhs_indices, rhs_contract, rhs_batch),
        ):
            for axis in range(len(mapping)):
                if axis not in contract and axis not in batch:
                    mapping[axis] = next(free)
        for la, ra in zip(lhs_contract, rhs_contract, strict=True):
            if lhs.aval.shape[la] != rhs.aval.shape[ra]:
                raise Unsupported("contraction dimension mismatch")
            bound = sp.Dummy("contract", integer=True)
            lhs_indices[la] = rhs_indices[ra] = bound
            limits.append((bound, 0, lhs.aval.shape[la] - 1))
        product = self.read(lhs, tuple(lhs_indices)) * self.read(rhs, tuple(rhs_indices))
        return sp.Sum(product, *limits) if limits else product

    def outputs(self):
        outputs = []
        for var in self.graph.jaxpr.outvars:
            indices = tuple(sp.Symbol(f"out{axis}", integer=True) for axis in range(var.aval.ndim))
            expression = self.read(var, indices)
            outputs.append(sp.factor_terms(sp.expand(expression)).as_dummy())
        return outputs


def trace(function, args):
    graph, output = jax.make_jaxpr(function, return_shape=True)(*args)
    leaves, tree = jax.tree.flatten(output)
    contract = [(tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves]
    return graph, (str(tree), contract)


def compare(name, left, right, args, *, negative=False):
    left_graph, left_contract = trace(left, args)
    right_graph, right_contract = trace(right, args)
    report = {
        "case": name,
        "input_shapes": [arg.shape for arg in args],
        "output_contract": left_contract,
        "primitives": [
            dict(Counter(eq.primitive.name for eq in g.jaxpr.eqns))
            for g in (left_graph, right_graph)
        ],
        "jaxprs": [str(left_graph), str(right_graph)],
    }
    if left_contract != right_contract:
        return report | {"status": "unknown", "reason": "output contract mismatch"}
    try:
        lhs, rhs = Lowering(left_graph).outputs(), Lowering(right_graph).outputs()
    except Unsupported as exc:
        return report | {"status": "unknown", "reason": str(exc)}
    report["indexed_outputs"] = [[str(expr) for expr in expressions] for expressions in (lhs, rhs)]
    if lhs == rhs:
        return report | {"status": "equivalent", "evidence": "identical exact indexed normal forms"}
    if negative:
        # A uniform symbolic tensor family is a legal subdomain of these inputs.
        # After substitution, Sum.doit closes constant sums; no tensor is allocated.
        parameters = {}

        def uniform(element):
            name = str(element.base.label)
            return parameters.setdefault(name, sp.Symbol(f"uniform_{name}", real=True))

        residual = (lhs[0] - rhs[0]).replace(lambda item: isinstance(item, sp.Indexed), uniform)
        residual = sp.expand(residual.doit())
        if residual.has(sp.Sum):
            return report | {"status": "unknown", "reason": "uniform family sum remained"}
        if residual == 0:
            return report | {"status": "unknown", "reason": "uniform family did not distinguish"}
        if residual.is_Rational and residual != 0:
            return report | {
                "status": "inequivalent",
                "evidence": f"uniform family yields exact nonzero constant {residual}",
            }
        polynomial = sp.Poly(residual, *sorted(residual.free_symbols, key=str), domain=sp.QQ)
        if not polynomial.is_zero:
            return report | {
                "status": "inequivalent",
                "evidence": f"uniform symbolic input family yields nonzero polynomial {residual}",
            }
    return report | {"status": "unknown", "reason": "different normal forms without evidence"}


def run():
    p = layer.PROFILES["glm"]

    def abstract(shape):
        return jax.ShapeDtypeStruct(shape, jnp.float32)

    qk_args = (
        abstract((p.batch, p.heads, p.latent + p.rope)),
        abstract((p.batch, p.top_k, p.latent + p.rope)),
    )
    pv_args = (
        abstract((p.batch, p.heads, p.top_k)),
        abstract((p.batch, p.top_k, p.latent)),
    )
    small = (abstract((2, 2)),) * 3
    results = [
        compare(
            "glm_qk_scale",
            lambda q, k: layer._qk(q, k, layer.SCALE),
            lambda q, k: layer._qk_scale_first(q, k, layer.SCALE),
            qk_args,
        ),
        compare("glm_pv_reduce", layer._pv, layer._pv_reduce, pv_args),
        compare(
            "glm_qk_missing_scale",
            lambda q, k: layer._qk(q, k, layer.SCALE),
            lambda q, k: layer._qk_without_scale(q, k, layer.SCALE),
            qk_args,
            negative=True,
        ),
        compare(
            "distributivity", lambda x, y, z: (x + y) * z, lambda x, y, z: x * z + y * z, small
        ),
        compare("unsupported_exp", jnp.exp, lambda x: x, (abstract((2, 2)),)),
        compare("empty_domain", lambda x: x, lambda x: 2 * x, (abstract((0,)),), negative=True),
        compare(
            "constant_difference",
            lambda x: jnp.float32(1),
            lambda x: jnp.float32(2),
            (abstract(()),),
            negative=True,
        ),
    ]
    expected = [
        "equivalent",
        "equivalent",
        "inequivalent",
        "equivalent",
        "unknown",
        "unknown",
        "inequivalent",
    ]
    assert [result["status"] for result in results] == expected
    return {
        "jax_version": jax.__version__,
        "sympy_version": sp.__version__,
        "adapter": "handwritten restricted jaxpr-to-indexed-expression translator",
        "assumptions": [
            "finite real interpretation of float32 operations",
            "no dtype-changing casts",
        ],
        "scope": "actual GLM helpers at glm profile shapes; not complete DSA/MLA layer",
        "compiled": False,
        "executed": False,
        "results": results,
    }


def main():
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
        patch.object(sp.Expr, "equals", forbidden),
        patch.object(sp.Expr, "_random", forbidden),
    ):
        report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
