"""Shape-preserving SymPy construction for the locally maintained jax2sympy front end."""

from dataclasses import dataclass, replace
from itertools import product
from math import prod

import numpy as np
import sympy as sp
from jax import dtypes, lax

from ._vendor.jax2sympy.primitive_mapping import primitive_to_sympy_op
from ._vendor.jax2sympy.translate import exact_array, jaxpr_to_sympy_expressions
from .algebra_finiteness import Finiteness, Source
from .algebra_ir import Budget
from .errors import UnsupportedJaxprError


def coordinates(shape):
    return tuple(
        sp.S.Zero if size == 1 else sp.Symbol(f"_i{axis}", integer=True, nonnegative=True)
        for axis, size in enumerate(shape)
    )


def dtype_kind(dtype):
    return "f" if dtypes.issubdtype(dtype, np.floating) else np.dtype(dtype).kind


def rename_bounds(expression, *, fresh=False):
    """Respect nested lexical scopes when renaming Sum/Product variables.

    SymPy 1.14 as_dummy reuses _0 inside nested sums and can capture outer
    references. Construction uses fresh Dummies; query keys use depth-qualified names.
    """
    cache = {}

    def visit(node, environment, depth):
        if node in environment:
            return environment[node]
        if not node.args:
            return node
        key = (node, tuple(environment.items()), depth)
        if key in cache:
            return cache[key]
        if isinstance(node, (sp.Sum, sp.Product)):
            bound = dict(environment)
            limits = []
            for position, (variable, start, stop) in enumerate(node.limits):
                symbol = (
                    sp.Dummy("bound", integer=True)
                    if fresh
                    else sp.Symbol(f"_bound_{depth}_{position}", integer=True)
                )
                bound[variable] = symbol
                limits.append(
                    (symbol, visit(start, environment, depth), visit(stop, environment, depth))
                )
            result = node.func(visit(node.function, bound, depth + 1), *limits)
        else:
            result = node.func(*(visit(arg, environment, depth) for arg in node.args))
        cache[key] = result
        return result

    return visit(expression, {}, 0)


def normalize_coordinates(expression, shape):
    """Reuse exact coordinate adaptation without invoking the legacy value normalizer."""
    from .algebra_indices import add, divide, integer, scale
    from .algebra_ir import Index

    symbols = {
        symbol: Index("free", axis) for axis, symbol in enumerate(coordinates(shape)) if symbol != 0
    }
    domains = {index: (0, shape[index.number]) for index in symbols.values()}
    for node in expression.atoms(sp.Sum, sp.Product):
        for symbol, start, stop in node.limits:
            if start.is_Integer and stop.is_Integer and symbol not in symbols:
                index = Index("sum", len(symbols))
                symbols[symbol] = index
                domains[index] = (int(start), int(stop) + 1)
    reverse = {value: key for key, value in symbols.items()}

    def parse(value):
        if value in symbols:
            return symbols[value]
        if value.is_Integer:
            return integer(int(value))
        if value.is_Add:
            return add(*(parse(child) for child in value.args))
        if value.is_Mul:
            coefficient, rest = value.as_coeff_Mul()
            if coefficient.is_Integer:
                return scale(parse(rest), int(coefficient))
        if value.func == sp.Mod and value.args[1].is_Integer:
            return divide(parse(value.args[0]), int(value.args[1]), domains, modulo=True)
        if value.func == sp.floor:
            numerator, denominator = value.args[0].as_numer_denom()
            if denominator.is_Integer:
                return divide(parse(numerator), int(denominator), domains)
        raise ValueError("not a supported coordinate expression")

    def emit(index):
        if index in reverse:
            return reverse[index]
        if index.kind in ("zero", "const"):
            return sp.Integer(index.number)
        if index.kind == "add":
            return sp.Add(*(emit(child) for child in index.args))
        if index.kind == "scale":
            return index.number * emit(index.args[0])
        if index.kind == "div":
            return sp.floor(emit(index.args[0]) / index.number)
        if index.kind == "mod":
            return sp.Mod(emit(index.args[0]), index.number)
        raise ValueError("unknown coordinate kind")

    replacements = {}
    for read in expression.atoms(sp.Indexed):
        try:
            replacements[read] = read.base[tuple(emit(parse(i)) for i in read.indices)]
        except ValueError:
            continue
    return expression.xreplace(replacements)


@dataclass(frozen=True)
class SymbolicTensor:
    shape: tuple
    dtype: object
    expression: sp.Basic
    defined: sp.Basic = sp.true
    finite: bool = True
    source: object = None

    def read(self, indices):
        replacements = dict(zip(coordinates(self.shape), map(sp.sympify, indices), strict=True))
        replacements.pop(sp.S.Zero, None)
        return (
            rename_bounds(self.expression, fresh=True).xreplace(replacements),
            rename_bounds(self.defined, fresh=True).xreplace(replacements),
        )


SCALAR = frozenset(
    "add sub mul div neg pow integer_pow sign square abs exp log sin cos tan asin "
    "acos atan sinh cosh tanh sqrt eq ne lt le gt ge not and or max min "
    "logaddexp atan2 add_any".split()
)


class SymbolicBuilder:
    """Construct values and conditions; all algebraic decisions belong to SymPy."""

    def __init__(self, budget=10000):
        self.budget = budget if isinstance(budget, Budget) else Budget(budget)
        self.rules = set()
        self.constants = []
        self.next_bound = 0
        from .algebra_lower import Lowerer

        self.finiteness = Finiteness(Lowerer(self.budget))
        self.definitions = {}
        self.canonical_values = {}

    def input(self, aval, position):
        kind = dtype_kind(aval.dtype)
        if kind not in "fbiu":
            raise UnsupportedJaxprError("algebra.dtype", "requires real, integer or Boolean inputs")
        assumptions = {"real": True} if kind == "f" else {"integer": True} if kind in "iu" else {}
        shape = tuple(aval.shape)
        symbol = sp.Symbol(f"x{position}", **assumptions)
        expression = (
            sp.IndexedBase(symbol, shape=shape, **assumptions)[coordinates(shape)]
            if shape
            else symbol
        )
        if kind == "b":
            expression = sp.Eq(expression, 1)
        return SymbolicTensor(
            shape, aval.dtype, expression, source=Source(None, shape, aval.dtype, finite=True)
        )

    def constant(self, value, aval):
        array = np.asarray(value, dtype=aval.dtype)
        if dtype_kind(aval.dtype) not in "fbiu":
            raise UnsupportedJaxprError("algebra.constant", "non-real constants are unsupported")
        if dtype_kind(aval.dtype) == "f" and (
            np.any(np.isnan(array)) or np.any(np.isposinf(array))
        ):
            raise UnsupportedJaxprError(
                "algebra.constant", "positive infinity and NaN literals are not represented"
            )
        self.budget.spend(max(1, array.size), path="algebra.constants")
        exact = exact_array(array)
        if array.size == 0:
            expression = sp.S.Zero
        elif array.size == 1 or np.all(array == array.flat[0]):
            expression = sp.sympify(exact.flat[0])
        else:
            # Immutable explicit constants retain their values, never fresh symbolic identities.
            indices = coordinates(array.shape)
            expression = sp.Piecewise(
                *[
                    (
                        exact[index],
                        sp.And(*(sp.Eq(i, j) for i, j in zip(indices, index, strict=True))),
                    )
                    for index in np.ndindex(array.shape)
                ]
            )
        finite = bool(np.all(np.isfinite(array)))
        if not finite:
            self.rules.add("negative_infinity_constant")
        source = Source(
            "constant",
            tuple(aval.shape),
            aval.dtype,
            finite=finite,
            constant=array.item() if array.shape == () else None,
        )
        return SymbolicTensor(
            tuple(aval.shape), aval.dtype, expression, finite=finite, source=source
        )

    def graph(self, graph, inputs=None):
        bindings = None if inputs is None else dict(zip(graph.invars, inputs, strict=True))
        return jaxpr_to_sympy_expressions(graph, var2sym=bindings, builder=self)[0]

    def finish(self, outputs, unused):
        from .algebra_trusted import _domain

        condition = sp.And(*(value.defined for value in unused))
        if condition != sp.true:
            self.budget.spend(path="algebra.dead.conditions")
            condition = _domain(condition, self.budget)
        if not outputs and condition != sp.true:
            raise UnsupportedJaxprError(
                "algebra.dead", "unused partial computation with no outputs"
            )
        return [replace(value, defined=sp.And(value.defined, condition)) for value in outputs]

    def definition(self, operation, values, outputs):
        """Intern exact tensor definitions; scalar SymPy functions carry only output indices."""
        from sympy.core.function import AppliedUndef

        from .algebra_trusted import _domain, simplify

        captured = set()
        for value in values:
            local = set(coordinates(value.shape))
            for symbol in value.expression.free_symbols | value.defined.free_symbols:
                if (
                    isinstance(symbol, sp.Symbol)
                    and symbol not in local
                    and (symbol.is_Dummy or symbol.is_integer)
                ):
                    captured.add(symbol)
        captured = tuple(sorted(captured, key=sp.default_sort_key))
        replacements = {
            s: sp.Symbol(f"_capture{i}", **({"integer": True} if s.is_integer else {"real": True}))
            for i, s in enumerate(captured)
        }
        signature = []
        for value in values:
            expression = value.expression
            cache_key = (expression, value.shape)
            if cache_key not in self.canonical_values:
                bounds = sp.And(
                    *(i < n for i, n in zip(coordinates(value.shape), value.shape, strict=True))
                )
                opaque_guard = any(
                    condition.has(AppliedUndef)
                    for pw in expression.atoms(sp.Piecewise)
                    for _, condition in pw.args
                )
                self.canonical_values[cache_key] = (
                    rename_bounds(expression)
                    if dtype_kind(value.dtype) in "biu" or opaque_guard
                    else simplify(expression, self.budget, assumptions=bounds)
                )
            canonical = self.canonical_values[cache_key].xreplace(replacements)
            if operation[0] == "convert_element_type":
                selected, count = [], 1
                for axis, extent in enumerate(value.shape):
                    if 1 < extent and count * extent <= 64:
                        selected.append((coordinates(value.shape)[axis], extent))
                        count *= extent
                if selected:
                    self.budget.spend(count, path="algebra.cast.coordinates")
                    canonical = tuple(
                        canonical.xreplace(dict(zip((i for i, _ in selected), point, strict=True)))
                        for point in product(*(range(n) for _, n in selected))
                    )
            signature.append(
                (
                    value.shape,
                    str(value.dtype),
                    canonical,
                    rename_bounds(_domain(value.defined, self.budget)).xreplace(replacements),
                )
            )
        key = (
            operation,
            tuple(signature),
            tuple((tuple(v.aval.shape), str(v.aval.dtype)) for v in outputs),
        )
        if key not in self.definitions:
            self.definitions[key] = len(self.definitions)
        return self.definitions[key], captured

    def bound(self, extent):
        symbol = sp.Dummy(f"k{self.next_bound}", integer=True, nonnegative=True)
        self.next_bound += 1
        return symbol, 0, extent - 1

    def broadcast(self, value, shape):
        indices = coordinates(shape)[len(shape) - len(value.shape) :]
        return value.read(
            tuple(
                sp.S.Zero if size == 1 else i for size, i in zip(value.shape, indices, strict=True)
            )
        )

    def universal(self, condition, limits):
        if condition == sp.true or not limits:
            return condition
        count = prod(int(stop - start + 1) for _, start, stop in limits)
        if count <= 32:
            self.budget.spend(max(1, count), path="algebra.conditions")
            return sp.And(
                *(
                    condition.xreplace(
                        dict(zip((limit[0] for limit in limits), point, strict=True))
                    )
                    for point in product(
                        *(range(int(start), int(stop) + 1) for _, start, stop in limits)
                    )
                )
            )
        return sp.Eq(sp.Product(sp.Piecewise((1, condition), (0, True)), *limits), 1)

    def summation(self, expression, limits, defined=sp.true):
        """Use the existing zero-based coordinate convention for total rectangular domains."""
        singletons = {
            variable: sp.sympify(start) for variable, start, stop in limits if start == stop
        }
        expression = expression.xreplace(singletons)
        limits = tuple(limit for limit in limits if limit[1] != limit[2])
        if not limits:
            return expression
        node = sp.Sum(expression, *limits)
        if defined != sp.true or expression.has(sp.oo, -sp.oo, sp.nan, sp.zoo):
            return node
        for variable, _, _ in tuple(node.limits):
            offsets = set()
            for read in node.function.atoms(sp.Indexed):
                for coordinate in read.indices:
                    if coordinate.has(variable):
                        offset = coordinate - variable
                        if not offset.is_Integer or offset < 0:
                            offsets.add(None)
                        else:
                            offsets.add(int(offset))
            if len(offsets) == 1 and None not in offsets and next(iter(offsets)) > 0:
                offset = next(iter(offsets))
                node = node.change_index(variable, variable + offset)
        # Extension is valid only for total reads that stay inside the source array.
        for variable, start, stop in node.limits:
            if start == 0:
                continue
            if start < 0:
                return node
            for read in node.function.atoms(sp.Indexed):
                for coordinate, extent in zip(read.indices, read.base.shape, strict=True):
                    if coordinate.has(variable) and (coordinate != variable or stop >= extent):
                        return node
        choices = [
            ((1, (v, 0, stop)),) if start == 0 else ((1, (v, 0, stop)), (-1, (v, 0, start - 1)))
            for v, start, stop in node.limits
        ]
        self.budget.spend(prod(len(choice) for choice in choices), path="algebra.index_domains")
        return sp.Add(
            *(
                prod(sign for sign, _ in selection)
                * sp.Sum(node.function, *(limit for _, limit in selection))
                for selection in product(*choices)
            )
        )

    def operation(self, eq, values, position):
        """Keep the existing source-defined finiteness analysis at the conversion boundary."""
        from sympy.core.function import AppliedUndef

        name = eq.primitive.name
        region = name in (
            "jit",
            "pjit",
            "cond",
            "scan",
            "while",
            "custom_jvp_call",
            "custom_vjp_call",
        )
        sources = []
        for var in eq.outvars:
            inputs = tuple(
                v.source or Source(None, v.shape, v.dtype, finite=v.finite) for v in values
            )
            source = Source(
                getattr(lax, name + "_p", None),
                tuple(var.aval.shape),
                var.aval.dtype,
                inputs,
                eq.params,
            )
            source.finite = (
                self.finiteness._finite(source) if not region and len(eq.outvars) == 1 else False
            )
            if dtype_kind(var.aval.dtype) in "biu":
                source.finite = True
            sources.append(source)
        self.rules.update(self.finiteness.lowerer.rules)
        layouts = {
            "broadcast_in_dim",
            "reshape",
            "transpose",
            "squeeze",
            "slice",
            "rev",
            "tile",
            "split",
            "stop_gradient",
            "copy",
            "pad",
        }
        if name == "reduce_sum" and all(values[0].shape[axis] == 1 for axis in eq.params["axes"]):
            layouts.add(name)
        narrowing = (
            name == "convert_element_type"
            and values
            and values[0].finite
            and not sources[0].finite
            and dtype_kind(values[0].dtype) == dtype_kind(eq.outvars[0].aval.dtype) == "f"
            and bool(values[0].expression.free_symbols)
        )
        if narrowing:
            from .algebra_trusted import simplify

            value = values[0]
            bounds = sp.And(
                *(i < n for i, n in zip(coordinates(value.shape), value.shape, strict=True))
            )
            argument = simplify(value.expression, self.budget, assumptions=bounds)
            label = f"cast_{value.dtype}_to_{eq.outvars[0].aval.dtype}"
            domain = sp.And(value.defined, sp.Eq(sp.Function("defined_" + label)(argument), 1))
            self.rules.add("scalar:convert_element_type")
            return [
                SymbolicTensor(
                    value.shape,
                    eq.outvars[0].aval.dtype,
                    sp.Function(label, real=True)(argument),
                    domain,
                    True,
                    sources[0],
                )
            ]
        if (
            not region
            and name not in layouts
            and (
                any(not v.finite for v in values)
                or narrowing
                or name
                in (
                    "top_k",
                    "approx_top_k",
                    "reduce",
                    "gather",
                    "sort",
                    "scatter",
                    "scatter-add",
                    "dynamic_update_slice",
                    "rem",
                    "shift_left",
                    "shift_right_logical",
                    "shift_right_arithmetic",
                    "xor",
                )
                or (
                    name in SCALAR
                    and dtype_kind(eq.outvars[0].aval.dtype) in "iu"
                    and any(v.expression.free_symbols for v in values)
                )
                or (
                    name == "convert_element_type"
                    and values[0].dtype != eq.outvars[0].aval.dtype
                    and dtype_kind(values[0].dtype) in "iu"
                    and values[0].expression.free_symbols
                )
                or (
                    name == "select_n"
                    and (
                        dtype_kind(eq.outvars[0].aval.dtype) in "biu"
                        or values[0].expression.has(AppliedUndef)
                    )
                )
                or (
                    name == "convert_element_type"
                    and values[0].expression.has(AppliedUndef)
                    and dtype_kind(eq.outvars[0].aval.dtype) in "iu"
                )
            )
        ):
            return self.opaque(eq, values, sources)
        outputs = self._operation(eq, values, position)
        if region:
            return outputs
        return [
            replace(
                value,
                source=source,
                finite=all(v.finite for v in values) if name in layouts else value.finite,
            )
            for value, source in zip(outputs, sources, strict=True)
        ]

    def opaque(self, eq, values, sources):
        """Congruence of a complete operation, without applying real algebra to nonfinite values."""
        from .algebra_lower import encode

        name = eq.primitive.name
        self.rules.add(name)
        identity, captured = self.definition((name, encode(eq.params)), values, eq.outvars)
        result = []
        for port, (var, source) in enumerate(zip(eq.outvars, sources, strict=True)):
            shape = tuple(var.aval.shape)
            args = (*captured, *coordinates(shape))
            label = f"{name}_{identity}_{port}"
            expression = sp.Function("opaque_" + label, real=True if source.finite else None)(*args)
            if dtype_kind(var.aval.dtype) == "b":
                expression = sp.Eq(expression, 1)
            domain = sp.Eq(sp.Function("defined_" + label)(*args), 1)
            result.append(
                SymbolicTensor(shape, var.aval.dtype, expression, domain, source.finite, source)
            )
        return result

    def _operation(self, eq, values, position):
        name, params = eq.primitive.name, eq.params
        path = f"algebra.convert.eqns[{position}].{name}"
        self.rules.add(name)
        if "call_jaxpr" in params and name in ("custom_jvp_call", "custom_vjp_call"):
            return self.graph(params["call_jaxpr"], values)
        if name in ("jit", "pjit"):
            if (
                "common_call" in params
                and all(value.finite for value in values)
                and all(dtype_kind(v.aval.dtype) == "f" for v in eq.outvars)
            ):
                identity, inputs, outputs = params["common_call"]
                identity, captured = self.definition(
                    ("common_jit", identity),
                    [values[i] for i in inputs],
                    [eq.outvars[i] for i in outputs],
                )
                domain = sp.Eq(sp.Function(f"defined_call_{identity}")(*captured), 1)
                result = [None] * len(outputs)
                for port, mapped in enumerate(outputs):
                    aval = eq.outvars[mapped].aval
                    if dtype_kind(aval.dtype) != "f":
                        raise UnsupportedJaxprError(path, "abstract call needs real-valued outputs")
                    expression = sp.Function(f"call_{identity}_{port}", real=True)(
                        *captured, *coordinates(aval.shape)
                    )
                    result[mapped] = SymbolicTensor(
                        tuple(aval.shape), aval.dtype, expression, domain
                    )
                self.rules.add("common_jit")
                return result
            return self.graph(params["jaxpr"], values)
        if name in ("cond", "scan", "while"):
            from .algebra_symbolic_regions import region

            return region(self, eq, values, path)
        if len(eq.outvars) != 1 and name != "split":
            raise UnsupportedJaxprError(path, "multiple-output primitive has no symbolic adapter")
        aval = eq.outvars[0].aval
        shape, axes = tuple(aval.shape), coordinates(aval.shape)

        def result(expression, defined=sp.true):
            return [
                SymbolicTensor(
                    shape,
                    aval.dtype,
                    normalize_coordinates(sp.sympify(expression), shape),
                    normalize_coordinates(sp.sympify(defined), shape),
                    finite=not sp.sympify(expression).has(sp.oo, -sp.oo, sp.nan, sp.zoo),
                )
            ]

        if name == "empty2":
            # The pinned JAX primitive's _empty2_lower emits a scalar zero,
            # including residual placeholders introduced by transpose rules.
            return result(0)

        if name in SCALAR:
            if params.get("accuracy") is not None:
                raise UnsupportedJaxprError(path, "non-default scalar accuracy")
            args, conditions = zip(*(self.broadcast(v, shape) for v in values), strict=True)
            if name in ("not", "and", "or") and any(dtype_kind(v.dtype) != "b" for v in values):
                raise UnsupportedJaxprError(path, "integer bitwise arithmetic is not Boolean logic")
            if (
                dtype_kind(aval.dtype) in "iu"
                and name not in ("max", "min")
                and any(arg.free_symbols for arg in args)
            ):
                raise UnsupportedJaxprError(
                    path, "fixed-width integer arithmetic has no real interpretation"
                )
            domain = sp.And(*conditions)
            if name == "div":
                domain &= sp.Ne(args[1], 0)
            elif name == "log":
                domain &= args[0] > 0
            elif name == "sqrt":
                domain &= args[0] >= 0
            elif name == "integer_pow" and params["y"] < 0:
                domain &= sp.Ne(args[0], 0)
            elif name == "pow":
                domain &= args[0] > 0
            elif name in ("asin", "acos"):
                domain &= sp.And(args[0] >= -1, args[0] <= 1)
            elif name == "tan":
                domain &= sp.Ne(sp.cos(args[0]), 0)
            elif name == "atan2":
                domain &= sp.Or(sp.Ne(args[0], 0), sp.Ne(args[1], 0))
            expression = primitive_to_sympy_op[name](
                [np.asarray(x, dtype=object) for x in args], eq
            )
            expression = sp.sympify(np.asarray(expression).item())
            if dtype_kind(aval.dtype) in "iu" and not expression.free_symbols:
                limits = np.iinfo(aval.dtype)
                if not expression.is_Integer or not limits.min <= expression <= limits.max:
                    raise UnsupportedJaxprError(
                        path, "integer constant arithmetic overflows its dtype"
                    )
            return result(expression, domain)
        if name in ("stop_gradient", "copy"):
            return result(*values[0].read(axes))
        if name == "convert_element_type":
            value = values[0]
            if np.dtype(value.dtype) == np.dtype(aval.dtype):
                return result(*value.read(axes))
            expression, domain = value.read(axes)
            if np.dtype(value.dtype).kind == "b":
                return result(sp.Piecewise((1, expression), (0, True)), domain)
            if not expression.free_symbols:
                converted = np.asarray(float(expression), dtype=aval.dtype)
                return result(exact_array(converted).item(), domain)
            if dtype_kind(value.dtype) == dtype_kind(aval.dtype) == "f":
                function = sp.Function(f"cast_{value.dtype}_to_{aval.dtype}", real=True)
                self.rules.add("scalar:convert_element_type")
                return result(function(expression), domain)
            raise UnsupportedJaxprError(
                path, "value-changing cast requires an explicit mathematical adapter"
            )
        if name == "broadcast_in_dim":
            indices = tuple(
                0 if size == 1 else axes[axis]
                for size, axis in zip(values[0].shape, params["broadcast_dimensions"], strict=True)
            )
            return result(*values[0].read(indices))
        if name == "transpose":
            indices = [None] * len(axes)
            for axis, source in enumerate(params["permutation"]):
                indices[source] = axes[axis]
            return result(*values[0].read(indices))
        if name == "squeeze":
            iterator = iter(axes)
            return result(
                *values[0].read(
                    tuple(
                        0 if i in params["dimensions"] else next(iterator)
                        for i in range(len(values[0].shape))
                    )
                )
            )
        if name == "reshape":
            original = values[0]
            if prod(shape) == 0:
                return result(0)
            order = params.get("dimensions") or tuple(range(len(original.shape)))
            old_shape = tuple(original.shape[i] for i in order)
            flat = sum(i * prod(shape[axis + 1 :]) for axis, i in enumerate(axes))
            indices = [None] * len(old_shape)
            for axis, source in enumerate(order):
                extent, stride = old_shape[axis], prod(old_shape[axis + 1 :])
                indices[source] = 0 if extent <= 1 else sp.Mod(sp.floor(flat / stride), extent)
            return result(*original.read(indices))
        if name == "slice":
            strides = params.get("strides") or (1,) * len(axes)
            return result(
                *values[0].read(
                    tuple(
                        start + i * stride
                        for start, i, stride in zip(
                            params["start_indices"], axes, strides, strict=True
                        )
                    )
                )
            )
        if name == "pad":
            original, padding = values
            indices, guards = [], []
            for coordinate, extent, (low, _, interior) in zip(
                axes, original.shape, params["padding_config"], strict=True
            ):
                shifted = coordinate - low
                indices.append(sp.floor(shifted / (interior + 1)))
                guards.extend(
                    (
                        shifted >= 0,
                        shifted < extent * (interior + 1),
                        sp.Eq(sp.Mod(shifted, interior + 1), 0),
                    )
                )
            guard = sp.And(*guards)
            expression, domain = original.read(indices)
            fill, fill_domain = padding.read(())
            return result(
                sp.Piecewise((expression, guard), (fill, True)),
                sp.Or(sp.And(guard, domain), sp.And(sp.Not(guard), fill_domain)),
            )
        if name == "dynamic_slice":
            indices = tuple(
                i + sp.Min(sp.Max(start.expression, 0), size - extent)
                for i, start, size, extent in zip(
                    axes, values[1:], values[0].shape, shape, strict=True
                )
            )
            expression, domain = values[0].read(indices)
            return result(expression, sp.And(domain, *(v.defined for v in values[1:])))
        if name == "rev":
            return result(
                *values[0].read(
                    tuple(
                        size - 1 - i if axis in params["dimensions"] else i
                        for axis, (size, i) in enumerate(zip(shape, axes, strict=True))
                    )
                )
            )
        if name == "tile":
            padded = (1,) * (len(shape) - len(values[0].shape)) + values[0].shape
            indices = tuple(
                0 if size == 1 else sp.Mod(i, size) for size, i in zip(padded, axes, strict=True)
            )
            return result(*values[0].read(indices[len(padded) - len(values[0].shape) :]))
        if name == "iota":
            return result(axes[params["dimension"]])
        if name in (
            "reduce_sum",
            "reduce_prod",
            "reduce_max",
            "reduce_min",
            "reduce_or",
            "reduce_and",
        ):
            original = values[0]
            limits = [self.bound(original.shape[axis]) for axis in params["axes"]]
            reductions = dict(zip(params["axes"], limits, strict=True))
            free = iter(axes)
            indices = tuple(
                reductions[i][0] if i in reductions else next(free)
                for i in range(len(original.shape))
            )
            expression, domain = original.read(indices)
            empty = any(original.shape[axis] == 0 for axis in params["axes"])
            if empty:
                return result(
                    {
                        "reduce_sum": 0,
                        "reduce_prod": 1,
                        "reduce_max": -sp.oo,
                        "reduce_min": sp.oo,
                        "reduce_or": sp.false,
                        "reduce_and": sp.true,
                    }[name]
                )
            if not limits:
                return result(expression, domain)
            if name in ("reduce_sum", "reduce_prod"):
                function = sp.Sum if name == "reduce_sum" else sp.Product
                return result(
                    self.summation(expression, limits, domain)
                    if name == "reduce_sum"
                    else function(expression, *limits),
                    self.universal(domain, limits),
                )
            count = prod(stop + 1 for _, _, stop in limits)
            self.budget.spend(count, path=path)
            terms = [
                expression.xreplace(dict(zip((lim[0] for lim in limits), point, strict=True)))
                for point in product(*(range(stop + 1) for _, _, stop in limits))
            ]
            function = {
                "reduce_max": sp.Max,
                "reduce_min": sp.Min,
                "reduce_or": sp.Or,
                "reduce_and": sp.And,
            }[name]
            return result(function(*terms), self.universal(domain, limits))
        if name == "dot_general":
            if params.get("precision") not in (None, (None, None)):
                # Explicit precision remains outside this ideal-real adapter's contract.
                raise UnsupportedJaxprError(path, "non-default dot precision")
            left, right = values
            (ca, cb), (ba, bb) = params["dimension_numbers"]
            ia, ib = [None] * len(left.shape), [None] * len(right.shape)
            free = iter(axes)
            for a, b in zip(ba, bb, strict=True):
                ia[a] = ib[b] = next(free)
            for indices, contracted, batched in ((ia, ca, ba), (ib, cb, bb)):
                for axis in range(len(indices)):
                    if axis not in contracted + batched:
                        indices[axis] = next(free)
            limits = []
            for a, b in zip(ca, cb, strict=True):
                limit = self.bound(left.shape[a])
                limits.append(limit)
                ia[a] = ib[b] = limit[0]
            a, da = left.read(ia)
            b, db = right.read(ib)
            if any(left.shape[axis] == 0 for axis in ca):
                return result(0)
            return result(
                self.summation(a * b, limits, sp.And(da, db)),
                self.universal(sp.And(da, db), limits),
            )
        if name == "select_n":
            pred, pred_domain = self.broadcast(values[0], shape)
            branches = [self.broadcast(value, shape) for value in values[1:]]
            boolean = np.dtype(values[0].dtype).kind == "b"
            guards = [
                (sp.Not(pred) if i == 0 else pred) if boolean else sp.Eq(pred, i)
                for i in range(len(branches))
            ]
            expression = sp.Piecewise(
                *[(value, guard) for (value, _), guard in zip(branches, guards, strict=True)],
                (sp.false if dtype_kind(aval.dtype) == "b" else sp.S.Zero, True),
            )
            domain = sp.And(
                pred_domain,
                sp.Or(
                    *[
                        sp.And(guard, domain)
                        for (_, domain), guard in zip(branches, guards, strict=True)
                    ]
                ),
            )
            return result(expression, domain)
        if name in ("concatenate", "stack"):
            axis = params.get("dimension", params.get("axis"))
            branches, domains, offset = [], [], 0
            for value in values:
                index = list(axes)
                if name == "stack":
                    del index[axis]
                    extent = 1
                else:
                    index[axis] -= offset
                    extent = value.shape[axis]
                expression, domain = value.read(index)
                guard = sp.And(axes[axis] >= offset, axes[axis] < offset + extent)
                branches.append((expression, guard))
                domains.append(sp.And(guard, domain))
                offset += extent
            return result(
                sp.Piecewise(
                    *branches, (sp.false if dtype_kind(aval.dtype) == "b" else sp.S.Zero, True)
                ),
                sp.Or(*domains),
            )
        if name == "split":
            outputs, offset = [], 0
            for var, extent in zip(eq.outvars, params["sizes"], strict=True):
                index = list(coordinates(var.aval.shape))
                index[params["axis"]] += offset
                expression, domain = values[0].read(index)
                outputs.append(
                    SymbolicTensor(tuple(var.aval.shape), var.aval.dtype, expression, domain)
                )
                offset += extent
            return outputs
        raise UnsupportedJaxprError(path, "no complete symbolic adapter for primitive")
