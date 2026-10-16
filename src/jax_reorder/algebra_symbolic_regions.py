"""Bind ordered region arguments to the same SymPy builder as the surrounding graph."""

import sympy as sp

from .algebra_symbolic import SymbolicTensor, coordinates
from .errors import UnsupportedJaxprError


def independent_scan(builder, eq, values):
    """Express independent additive state updates as their defining finite sums."""
    params = eq.params
    nc, nk, length = params["num_consts"], params["num_carry"], params["length"]
    if any(value.defined != sp.true or not value.finite for value in values):
        return None
    time, _, _ = builder.bound(length)
    states, bases = [], []
    for index, initial in enumerate(values[nc : nc + nk]):
        if str(initial.dtype) not in ("float16", "bfloat16", "float32", "float64"):
            return None
        base = sp.Dummy(f"carry_{index}", real=True)
        expression = (
            sp.IndexedBase(base, shape=initial.shape, real=True)[coordinates(initial.shape)]
            if initial.shape
            else base
        )
        bases.append(base)
        states.append(SymbolicTensor(initial.shape, initial.dtype, expression))
    slices = []
    for value in values[nc + nk :]:
        expression, domain = value.read((time, *coordinates(value.shape[1:])))
        slices.append(
            SymbolicTensor(value.shape[1:], value.dtype, expression, domain, value.finite)
        )
    body = builder.graph(params["jaxpr"], [*values[:nc], *states, *slices])
    if any(
        value.defined != sp.true or value.expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
        for value in body
    ):
        return None
    fixed = {
        base: initial
        for base, state, updated, initial in zip(
            bases, states, body[:nk], values[nc : nc + nk], strict=True
        )
        if state.expression == updated.expression
    }

    def substitute(expression):
        replacements = {
            base: initial.expression for base, initial in fixed.items() if not initial.shape
        }
        for read in expression.atoms(sp.Indexed):
            if read.base.label in fixed:
                replacements[read] = fixed[read.base.label].read(read.indices)[0]
        return expression.xreplace(replacements)

    changing = [base for base in bases if base not in fixed]
    increments = []
    for state, updated in zip(states, body[:nk], strict=True):
        builder.budget.spend(path="algebra.scan.update")
        delta = sp.simplify(substitute(updated.expression - state.expression))
        if delta.has(*changing):
            return None
        increments.append(delta)
    if any(substitute(value.expression).has(*changing) for value in body[nk:]):
        return None
    outputs = []
    for initial, delta in zip(values[nc : nc + nk], increments, strict=True):
        expression = (
            initial.expression + builder.summation(delta, ((time, 0, length - 1),))
            if length
            else initial.expression
        )
        outputs.append(SymbolicTensor(initial.shape, initial.dtype, expression))
    for var, value in zip(eq.outvars[nk:], body[nk:], strict=True):
        axes = coordinates(var.aval.shape)
        expression = substitute(value.read(axes[1:])[0]).xreplace({time: axes[0]})
        outputs.append(SymbolicTensor(tuple(var.aval.shape), var.aval.dtype, expression))
    builder.rules.add("scan:independent_accumulation")
    return outputs


def region(builder, eq, values, path):
    params, name = eq.params, eq.primitive.name
    if name == "cond":
        selector = values[0]
        branches = [builder.graph(graph, values[1:]) for graph in params["branches"]]
        outputs = []
        for var, alternatives in zip(eq.outvars, zip(*branches, strict=True), strict=True):
            arms, domains = [], []
            for index, value in enumerate(alternatives):
                guard = sp.Eq(selector.expression, index)
                arms.append((value.expression, guard))
                domains.append(sp.And(guard, value.defined))
            outputs.append(
                SymbolicTensor(
                    tuple(var.aval.shape),
                    var.aval.dtype,
                    sp.Piecewise(
                        *arms, (sp.false if str(var.aval.dtype) == "bool" else sp.S.Zero, True)
                    ),
                    sp.And(selector.defined, sp.Or(*domains)),
                    all(value.finite for value in alternatives),
                )
            )
        return outputs
    if name == "scan":
        independent = independent_scan(builder, eq, values)
        if independent is not None:
            return independent
        count, length, ncarry = params["num_consts"], params["length"], params["num_carry"]
        constants = values[:count]
        carry = values[count : count + ncarry]
        sequences = values[count + ncarry :]
        # Unrolling is symbolic, explicitly budgeted, and never invokes the JAX loop.
        if length > 64:
            raise UnsupportedJaxprError(path, "scan length exceeds bounded symbolic unrolling")
        emissions = [[] for _ in eq.outvars[ncarry:]]
        order = reversed(range(length)) if params["reverse"] else range(length)
        for index in order:
            builder.budget.spend(path=path)
            slices = []
            for value in sequences:
                expression, domain = value.read((sp.Integer(index), *coordinates(value.shape[1:])))
                slices.append(
                    SymbolicTensor(value.shape[1:], value.dtype, expression, domain, value.finite)
                )
            outputs = builder.graph(params["jaxpr"], [*constants, *carry, *slices])
            carry = outputs[:ncarry]
            for target, value in zip(emissions, outputs[ncarry:], strict=True):
                target.append((index, value))
        outputs = list(carry)
        for var, emitted in zip(eq.outvars[ncarry:], emissions, strict=True):
            axes = coordinates(var.aval.shape)
            arms, domains = [], []
            for index, value in emitted:
                expression, domain = value.read(axes[1:])
                guard = sp.Eq(axes[0], index)
                arms.append((expression, guard))
                domains.append(sp.And(guard, domain))
            outputs.append(
                SymbolicTensor(
                    tuple(var.aval.shape),
                    var.aval.dtype,
                    sp.Piecewise(
                        *arms, (sp.false if str(var.aval.dtype) == "bool" else sp.S.Zero, True)
                    )
                    if arms
                    else sp.S.Zero,
                    sp.Or(*domains) if arms else sp.true,
                    all(value.finite for _, value in emitted),
                )
            )
        return outputs
    if name == "while":
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        condition_constants, body_constants = values[:nc], values[nc : nc + nb]
        carry = values[nc + nb :]
        for _ in range(65):
            predicate = builder.graph(params["cond_jaxpr"], [*condition_constants, *carry])[0]
            truth = sp.simplify(predicate.expression)
            if predicate.defined != sp.true:
                raise UnsupportedJaxprError(path, "loop predicate conditions are not established")
            if truth == sp.false:
                return carry
            if truth != sp.true:
                raise UnsupportedJaxprError(
                    path, "data-dependent loop termination is not represented"
                )
            builder.budget.spend(path=path)
            carry = builder.graph(params["body_jaxpr"], [*body_constants, *carry])
        raise UnsupportedJaxprError(path, "while loop exceeds bounded symbolic unrolling")
    raise UnsupportedJaxprError(path, "unsupported region")
