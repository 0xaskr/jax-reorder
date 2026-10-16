# ruff: noqa: E731, B007, B905
import numpy as np
import sympy as sy

# jax 0.8.0 introduced TypnedNdArrays
from jax.extend.core import Literal

from .primitive_mapping import primitive_to_sympy_op


def exact_array(value):
    def exact_scalar(x):
        x = np.asarray(x).item()
        if isinstance(x, (bool, np.bool_)):
            return sy.true if x else sy.false
        if isinstance(x, (int, np.integer)):
            return sy.Integer(int(x))
        if isinstance(x, (float, np.floating)) and np.isfinite(x):
            return sy.Rational(float(x))
        return sy.sympify(x)

    return np.vectorize(exact_scalar, otypes=[object])(np.asarray(value))


def get_sym_invar(invars, var2sym):
    inexprs = []
    for inv in invars:
        if isinstance(inv, Literal):
            inexprs.append(exact_array(inv.val))
        else:
            inexprs.append(var2sym[inv])
    return inexprs


def get_sym_outvar(inexprs, eqn):
    prim_name = eqn.primitive.name
    if prim_name not in primitive_to_sympy_op:
        raise NotImplementedError(f"No Sympy translation rule for JAX primitive '{prim_name}'.")
    out_expr = primitive_to_sympy_op[prim_name](inexprs, eqn)
    if not isinstance(out_expr, list):  # multi var outputs are put in a list
        return [out_expr]
    else:
        return out_expr


def jaxpr_to_sympy_expressions(jaxpr, var2sym=None, x_cnt=0, c_cnt=0, *, builder=None):
    if builder is not None:
        return _build_expressions(jaxpr, var2sym, x_cnt, c_cnt, builder)
    if var2sym is None:
        var2sym = {}

    def append_vars(vars, var2sym, sym_name, cnt):
        for i, var in enumerate(vars):  # go through all inputs to the jaxpr
            assert var not in var2sym  # double check we are not overwriting anything
            shape = var.aval.shape  # the abstract shape of the input
            num_elements = int(np.prod(shape))
            kind = np.dtype(var.aval.dtype).kind
            assumptions = (
                {"real": True} if kind == "f" else {"integer": True} if kind in "iu" else {}
            )
            symbols_list = sy.symbols(
                [f"{sym_name}{i + cnt}_{j}" for j in range(num_elements)], **assumptions
            )
            symbolic_array = np.array(symbols_list).reshape(shape)
            var2sym[var] = symbolic_array
        return var2sym, cnt + len(vars)

    def substitute_symbol(symbol, sym_map):
        return symbol.subs(sym_map)

    var2sym, x_cnt = append_vars(jaxpr.jaxpr.invars, var2sym, "x", cnt=x_cnt)
    for var, value in zip(jaxpr.jaxpr.constvars, jaxpr.consts, strict=True):
        var2sym[var] = exact_array(value)
    c_cnt += len(jaxpr.jaxpr.constvars)

    for i, eqn in enumerate(jaxpr.eqns):
        # print(f"eqn number: {i}")
        # print(f"eqn: {eqn}")
        if eqn.primitive.name in ("pjit", "jit"):
            _jaxpr = eqn.params.get("jaxpr", None)
            sub_syms_out, sub_var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(
                _jaxpr, var2sym={}, x_cnt=x_cnt, c_cnt=c_cnt
            )
            syms_in = get_sym_invar(eqn.invars, var2sym)
            iterator = iter(sub_var2sym)
            sub_syms_in = [sub_var2sym[next(iterator)] for _ in range(len(syms_in))]
            sym_map = {}
            for sym_in, sub_sym_in in zip(syms_in, sub_syms_in):
                sym_map.update({key: value for key, value in zip(sub_sym_in.flat, sym_in.flat)})
            sub_syms_out = [np.vectorize(substitute_symbol)(f, sym_map) for f in sub_syms_out]
            exprs = sub_syms_out
        else:
            inexprs = get_sym_invar(eqn.invars, var2sym)
            exprs = get_sym_outvar(inexprs, eqn)
        exprs = [
            np.asarray(expr) for expr in exprs
        ]  # check we dont lose the array with squeezing and indexing
        for outvar, expr in zip(eqn.outvars, exprs):
            # print(f"expression added to var2sym: {expr} under: {outvar}")
            assert outvar.aval.shape == expr.shape
            var2sym[outvar] = expr

    out_syms = get_sym_invar(jaxpr.jaxpr.outvars, var2sym)
    return out_syms, var2sym, x_cnt, c_cnt


def _build_expressions(graph, bindings, x_cnt, c_cnt, builder):
    """Reuse upstream SSA traversal with a shape-preserving mathematical builder."""
    environment = {} if bindings is None else dict(bindings)
    for position, var in enumerate(graph.invars):
        if var not in environment:
            environment[var] = builder.input(var.aval, position + x_cnt)
    for var, value in zip(graph.constvars, graph.consts, strict=True):
        environment[var] = builder.constant(value, var.aval)

    def read(var):
        if hasattr(var, "val"):
            return builder.constant(var.val, var.aval)
        return environment[var]

    for position, equation in enumerate(graph.eqns):
        builder.budget.spend(path=f"algebra.convert.eqns[{position}]")
        values = [read(var) for var in equation.invars]
        outputs = builder.operation(equation, values, position)
        for var, output in zip(equation.outvars, outputs, strict=True):
            if tuple(var.aval.shape) != output.shape:
                raise ValueError("symbolic output shape differs from the jaxpr")
            environment[var] = output
    used = set(var for var in graph.outvars if not hasattr(var, "val"))
    unused = []
    for equation in reversed(graph.eqns):
        if any(var in used for var in equation.outvars):
            used.update(var for var in equation.invars if not hasattr(var, "val"))
        else:
            unused.extend(environment[var] for var in equation.outvars)
    unused.extend(environment[var] for var in graph.invars if var not in used)
    return (
        builder.finish([read(var) for var in graph.outvars], unused),
        environment,
        x_cnt + len(graph.invars),
        c_cnt + len(graph.constvars),
    )
