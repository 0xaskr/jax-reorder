"""Exact representable input specializations for complete-output counterexamples.

The candidates are only searches for negative witnesses. Acceptance requires
proved in-bounds reads, true original domains and unequal exact scalar values;
matching candidates never provide an equivalence conclusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import sympy as sp
from jax import dtypes
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational

from ._typing import ArrayAval
from .budget import Budget
from .expression import _rebuild, dag_atoms, substitute

type _ExponentialPolynomial = dict[sp.Rational, sp.Rational]
type DomainCache = dict[
    tuple[
        tuple[tuple[tuple[int, ...], str], ...],
        tuple[tuple[str, tuple[str, str]], ...],
        str,
        sp.Basic,
    ],
    bool,
]


class _Unproved(Exception):
    pass


def _representable_float(value: sp.Basic, dtype: str) -> bool:
    """Certify an exact finite dtype representation by exact rational roundtrip."""
    if not value.is_Rational:
        return False
    try:
        with np.errstate(all="ignore"):
            converted = np.asarray(float(value), dtype=dtype).item()
        return bool(np.isfinite(converted)) and sp.Rational(float(converted)) == value
    except (ValueError, OverflowError):
        return False


def _exponential_equal(left: sp.Basic, right: sp.Basic, budget: Budget) -> bool | None:
    """Compare closed rational functions of exp(rational) without huge degrees.

    For a common positive denominator D, exp(1/D) is transcendental: its
    D-th power is e. Distinct rational exponents become distinct Laurent
    monomials in exp(1/D), so a rational linear combination is zero exactly
    when all coefficients vanish. This also certifies every divisor;
    there is no numerical threshold or treatment of exp(-1000000) as zero.
    """

    def add(a: _ExponentialPolynomial, b: _ExponentialPolynomial) -> _ExponentialPolynomial:
        result = dict(a)
        budget.spend(len(b), path="algebra.witness.exponentials")
        for exponent, coefficient in b.items():
            result[exponent] = result.get(exponent, sp.S.Zero) + coefficient
        return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}

    def multiply(a: _ExponentialPolynomial, b: _ExponentialPolynomial) -> _ExponentialPolynomial:
        budget.spend(len(a) * len(b), path="algebra.witness.exponentials")
        result: _ExponentialPolynomial = {}
        for x, coefficient_x in a.items():
            for y, coefficient_y in b.items():
                exponent = x + y
                result[exponent] = result.get(exponent, sp.S.Zero) + coefficient_x * coefficient_y
        return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}

    def power(a: _ExponentialPolynomial, exponent: int) -> _ExponentialPolynomial:
        result = {sp.S.Zero: sp.S.One}
        while exponent:
            if exponent & 1:
                result = multiply(result, a)
            exponent //= 2
            if exponent:
                a = multiply(a, a)
        return result

    cache: dict[sp.Basic, tuple[_ExponentialPolynomial, _ExponentialPolynomial]] = {}

    def fraction(node: sp.Basic) -> tuple[_ExponentialPolynomial, _ExponentialPolynomial]:
        if node in cache:
            return cache[node]
        budget.spend(path="algebra.witness.exponentials")
        one = {sp.S.Zero: sp.S.One}
        if node.is_Rational:
            result = ({sp.S.Zero: node} if node else {}, one)
        elif node is sp.E:
            result = ({sp.S.One: sp.S.One}, one)
        elif node.func is sp.exp and node.args[0].is_Rational:
            result = ({node.args[0]: sp.S.One}, one)
        elif node.is_Add:
            numerator, denominator = {}, one
            for term in node.args:
                a, b = fraction(term)
                numerator = add(multiply(numerator, b), multiply(a, denominator))
                denominator = multiply(denominator, b)
            result = numerator, denominator
        elif node.is_Mul:
            numerator = denominator = one
            for term in node.args:
                a, b = fraction(term)
                numerator, denominator = multiply(numerator, a), multiply(denominator, b)
            result = numerator, denominator
        elif node.is_Pow and node.args[1].is_Integer:
            numerator, denominator = fraction(node.args[0])
            exponent = int(node.args[1])
            if exponent < 0:
                if not numerator:
                    raise _Unproved
                numerator, denominator = denominator, numerator
                exponent = -exponent
            result = power(numerator, exponent), power(denominator, exponent)
        else:
            raise _Unproved
        cache[node] = result
        return result

    try:
        a, b = fraction(left)
        c, d = fraction(right)
        return multiply(a, d) == multiply(c, b)
    except _Unproved:
        return None


def _bounds(value: sp.Basic, ranges: dict[sp.Basic, tuple[int, int]]) -> tuple[int, int]:
    if value in ranges:
        return ranges[value]
    if value.is_Integer:
        return int(value), int(value)
    if value.is_Add:
        # Preserve the correlation in an expanded Euclidean remainder. Only
        # integer numerators and positive integer divisors justify this rule.
        for quotient in dag_atoms(value, sp.floor):
            numerator, denominator = quotient.args[0].as_numer_denom()
            coefficient = value.coeff(quotient)
            if (
                denominator.is_Integer
                and denominator > 0
                and numerator.is_integer is True
                and coefficient.is_Integer
                and coefficient < 0
                and coefficient % denominator == 0
            ):
                scale = -coefficient / denominator
                remainder = value - scale * (numerator - denominator * quotient)
                if not remainder.has(quotient):
                    lo, hi = _bounds(remainder, ranges)
                    a, b = _bounds(sp.Mod(numerator, denominator), ranges)
                    return lo + int(scale) * a, hi + int(scale) * b
        pairs = [_bounds(arg, ranges) for arg in value.args]
        return sum(a for a, _ in pairs), sum(b for _, b in pairs)
    if value.is_Mul:
        lo = hi = 1
        for arg in value.args:
            a, b = _bounds(arg, ranges)
            products = (lo * a, lo * b, hi * a, hi * b)
            lo, hi = min(products), max(products)
        return lo, hi
    if value.func is sp.Mod and value.args[1].is_Integer and value.args[1] > 0:
        a, b = _bounds(value.args[0], ranges)
        n = int(value.args[1])
        return (a % n, b % n) if a // n == b // n else (0, n - 1)
    if value.func is sp.floor:
        numerator, denominator = value.args[0].as_numer_denom()
        if denominator.is_Integer and denominator > 0:
            a, b = _bounds(numerator, ranges)
            return a // int(denominator), b // int(denominator)
    if isinstance(value, sp.Piecewise) and value.args[-1][1] == sp.true:
        intervals = [_bounds(branch, ranges) for branch, _ in value.args]
        return min(lo for lo, _ in intervals), max(hi for _, hi in intervals)
    raise _Unproved


def _constant_intervals(
    function: sp.Basic,
    variable: sp.Symbol,
    start: int,
    stop: int,
    budget: Budget,
    *,
    product: bool = False,
) -> sp.Basic:
    """Contract piecewise constants by affine guard cuts, never by lanes."""
    if not function.has(variable):
        count = stop - start + 1
        return function**count if product else function * count
    cuts = {start, stop + 1}
    guards = [guard for guard in dag_atoms(function, Relational) if guard.has(variable)]
    for guard in guards:
        _ExponentialPolynomial = sp.Poly(guard.lhs - guard.rhs, variable)
        if _ExponentialPolynomial.degree() != 1:
            raise _Unproved
        slope, offset = _ExponentialPolynomial.all_coeffs()
        if not slope.is_Rational or not offset.is_Rational:
            raise _Unproved
        root = int(sp.floor(-offset / slope))
        cuts.update(point for point in (root, root + 1) if start <= point <= stop + 1)
    budget.spend(len(cuts) * max(1, len(guards)), path="algebra.witness.intervals")
    result = []
    ordered = sorted(cuts)
    for lo, hi in zip(ordered[:-1], ordered[1:], strict=True):
        replacements = {}
        for guard in guards:
            truth = guard.xreplace({variable: sp.Integer(lo)})
            if truth not in (sp.true, sp.false) or truth != guard.xreplace(
                {variable: sp.Integer(hi - 1)}
            ):
                raise _Unproved
            replacements[guard] = truth
        value = substitute(function, replacements)
        if value.has(variable):
            raise _Unproved
        result.append(value ** (hi - lo) if product else (hi - lo) * value)
    return sp.Mul(*result) if product else sp.Add(*result)


def _bounded_truth(condition: Relational, ranges: dict[sp.Basic, tuple[int, int]]) -> sp.Basic:
    """Discharge only a relation proved throughout its current integer box."""
    try:
        lo, hi = _bounds(condition.lhs - condition.rhs, ranges)
    except _Unproved:
        return condition
    outcomes = {
        "==": (lo == hi == 0, hi < 0 or lo > 0),
        "!=": (hi < 0 or lo > 0, lo == hi == 0),
        "<": (hi < 0, lo >= 0),
        "<=": (hi <= 0, lo > 0),
        ">": (lo > 0, hi <= 0),
        ">=": (lo >= 0, hi < 0),
    }
    yes, no = outcomes[condition.rel_op]
    return sp.true if yes else sp.false if no else condition


def _boolean(function: type[sp.Basic], args: Sequence[sp.Basic]) -> sp.Basic:
    """Local complementary-operand identities, without distributive expansion."""
    if function in (sp.And, sp.Or):
        args = tuple(
            child for arg in args for child in (arg.args if arg.func == function else (arg,))
        )
        choices = set(args)
        opposite = sp.And if function == sp.Or else sp.Or
        if any(
            sp.Not(arg) in choices
            or isinstance(arg, opposite)
            and all(sp.Not(child) in choices for child in arg.args)
            for arg in args
        ):
            return sp.true if function == sp.Or else sp.false
    return sp.to_nnf(function(*args, evaluate=True), simplify=True)


def _coordinate_zero(
    value: sp.Basic, ranges: dict[sp.Basic, tuple[int, int]], budget: Budget
) -> sp.Basic:
    """Exact one-hot membership after specializing a possibly piecewise address."""
    condition = sp.Eq(value, 0)
    if isinstance(condition, Relational):
        condition = _bounded_truth(condition, ranges)
    if condition in (sp.true, sp.false):
        return condition
    choices = dag_atoms(value, sp.Piecewise)
    if not choices:
        return condition
    choice = min(choices, key=sp.default_sort_key)
    budget.spend(len(choice.args), path="algebra.witness.address_cases")
    covered, cases = sp.false, []
    for branch, guard in choice.args:
        active = _boolean(sp.And, (sp.Not(covered), guard))
        if active != sp.false:
            cases.append(
                _boolean(
                    sp.And,
                    (active, _coordinate_zero(substitute(value, {choice: branch}), ranges, budget)),
                )
            )
        covered = _boolean(sp.Or, (covered, guard))
    if covered != sp.true:
        raise _Unproved
    return _boolean(sp.Or, cases)


def exact_witness(
    left: sp.Basic,
    right: sp.Basic,
    domains: Sequence[sp.Basic],
    inputs: Sequence[ArrayAval],
    budget: Budget,
    casts: Mapping[str, tuple[str, str]] | None = None,
    *,
    finite_values: bool = False,
    domain_cache: DomainCache | None = None,
) -> tuple[str, sp.Basic, sp.Basic] | None:
    """Try exact input patterns; return only a certified unequal pair."""
    from .sympy_ops import equal

    expressions = (left, right, *domains)
    casts = {} if casts is None else casts
    domain_cache = {} if domain_cache is None else domain_cache
    domain_context = (
        tuple((tuple(value.shape), str(value.dtype)) for value in inputs),
        tuple(sorted(casts.items())),
    )
    if any(expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for expression in expressions) or any(
        node.func.__name__.removeprefix("defined_") not in casts
        for expression in expressions
        for node in dag_atoms(expression, AppliedUndef)
    ):
        return None
    signatures = {f"x{index}": tuple(value.shape) for index, value in enumerate(inputs)}
    boolean_inputs = {
        f"x{index}" for index, value in enumerate(inputs) if np.dtype(value.dtype).kind == "b"
    }
    floating_inputs = {
        f"x{index}"
        for index, value in enumerate(inputs)
        if dtypes.issubdtype(value.dtype, np.floating)
    }
    address_dependencies: dict[sp.Basic, bool] = {}
    coordinate_bounds: dict[
        tuple[sp.Basic, tuple[tuple[int, int], ...]], tuple[int, int] | None
    ] = {}
    coordinate_slots: list[sp.Symbol] = []
    checked_casts: dict[tuple[sp.Basic, str, str], bool] = {}

    def input_address(index: sp.Basic) -> bool:
        if index not in address_dependencies:
            address_dependencies[index] = index.has(sp.Indexed) or any(
                isinstance(symbol, sp.Symbol) and str(symbol) in signatures
                for symbol in index.free_symbols
            )
        return address_dependencies[index]

    def constant_coordinates(
        expression: sp.Basic, ranges: dict[sp.Basic, tuple[int, int]]
    ) -> sp.Basic:
        """Fold only coordinate quotients constant throughout this lexical box."""
        rewritten: dict[sp.Basic, sp.Basic] = {}

        def local(node: sp.Basic) -> sp.Basic:
            if node in rewritten:
                return rewritten[node]
            # An inner binder may shadow an outer symbol with another range.
            # Its own visit establishes the scope in which quotients are valid.
            if isinstance(node, (sp.Sum, sp.Product)):
                return node
            if isinstance(node, (Relational, sp.And, sp.Or, sp.Not, sp.ITE)) and (
                input_address(node) or node.has(AppliedUndef)
            ):
                # Rebuilding a data relation or Boolean expression can erase
                # unread operands, even when its arithmetic stays unevaluated.
                return node
            if (
                node.func in (sp.floor, sp.Mod)
                and not input_address(node)
                and not node.has(AppliedUndef, sp.Sum, sp.Product)
            ):
                captures = tuple(
                    (axis, interval)
                    for axis, interval in ranges.items()
                    if axis in node.free_symbols
                )
                while len(coordinate_slots) < len(captures):
                    coordinate_slots.append(sp.Dummy("witness_coordinate", integer=True))
                # Only the proof key is alpha-normalized. Different axes get
                # different slots, repeated occurrences retain their identity,
                # and every captured interval stays in the key.
                key = (
                    substitute(
                        node,
                        {
                            axis: coordinate_slots[position]
                            for position, (axis, _) in enumerate(captures)
                        },
                    ),
                    tuple(interval for _, interval in captures),
                )
                if key not in coordinate_bounds:
                    budget.spend(path="algebra.witness.coordinate_bounds")
                    try:
                        coordinate_bounds[key] = _bounds(node, ranges)
                    except _Unproved:
                        coordinate_bounds[key] = None
                interval = coordinate_bounds[key]
                if interval is not None and interval[0] == interval[1]:
                    result = sp.Integer(interval[0])
                    rewritten[node] = result
                    return result
            args = tuple(local(arg) for arg in node.args)
            if (
                args != node.args
                and isinstance(node, sp.Expr)
                and not isinstance(node, (sp.Indexed, sp.IndexedBase))
            ):
                # In particular, a proved zero coordinate must not erase an
                # unvisited read/cast through 0*x, sin(0*x), or another consumer.
                result = node.func(*args, evaluate=False)
            else:
                result = _rebuild(node, args)
            rewritten[node] = result
            return result

        return local(expression)

    # Isolating one input can remove an expensive shared nonlinear prefix.
    # This is only candidate ordering, never an equality or domain inference.
    patterns = ("one", "one_hot", "zero")
    value_casts = any(value.has(AppliedUndef) for value in (left, right))
    dynamic_addresses = any(
        input_address(index)
        for value in (left, right)
        for read in dag_atoms(value, sp.Indexed)
        for index in read.indices
    )
    if not value_casts and dynamic_addresses:
        # Uniform values cannot distinguish two legal input-selected addresses.
        patterns = ("one_hot", "one", "zero")
    if casts and dynamic_addresses:
        # A whole floating slice preserves equal scores along contracted
        # lanes while still distinguishing pages. Integer metadata is zero.
        # This can satisfy exact-cast domains that a single hot lane cannot.
        patterns = ("first_slice", *patterns)
    if casts and value_casts and len(inputs) >= 3:
        # Trailing integer metadata (such as a runtime page table) is not
        # the value operand isolated by this float-cast search heuristic.
        floating = [
            i for i, value in enumerate(inputs) if dtypes.issubdtype(value.dtype, np.floating)
        ]
        referenced_inputs = {
            str(read.base.label) for value in (left, right) for read in dag_atoms(value, sp.Indexed)
        } | {
            str(symbol)
            for value in (left, right)
            for symbol in value.free_symbols
            if isinstance(symbol, sp.Symbol)
        }
        if floating and f"x{floating[-1]}" in referenced_inputs:
            patterns = (f"only_input_{floating[-1]}", *patterns)
    for pattern in patterns:
        cache: dict[tuple[sp.Basic, tuple[tuple[sp.Basic, tuple[int, int]], ...]], sp.Basic] = {}
        checked_reads: set[tuple[sp.Basic, tuple[tuple[sp.Basic, tuple[int, int]], ...]]] = set()
        domains_validated = [False]

        def input_one(name: str, pattern: str = pattern) -> bool:
            return (
                pattern == "one"
                or pattern == f"only_input_{name.removeprefix('x')}"
                or pattern == "first_slice"
                and name in floating_inputs
            )

        def indexed_coordinates(
            node: sp.Indexed, ranges: dict[sp.Basic, tuple[int, int]]
        ) -> tuple[tuple[int, ...], tuple[sp.Basic, ...]]:
            shape = signatures.get(str(node.base.label))
            if shape is None or shape != tuple(node.base.shape):
                raise _Unproved
            # Address reads belong to the same candidate input as payload
            # reads. Evaluate and validate them recursively before checking
            # the resulting address of this outer read.
            indices = tuple(
                visit(index, ranges) if input_address(index) else index for index in node.indices
            )
            for index, extent in zip(indices, shape, strict=True):
                lo, hi = _bounds(index, ranges)
                if not 0 <= lo <= hi < extent:
                    raise _Unproved
            return shape, indices

        def read_bounds(
            node: sp.Basic,
            ranges: dict[sp.Basic, tuple[int, int]],
            checked: set[
                tuple[sp.Basic, tuple[tuple[sp.Basic, tuple[int, int]], ...]]
            ] = checked_reads,
        ) -> None:
            """Keep every read obligation when finite value arithmetic is skipped.

            This is a memoized metadata walk, not scalar evaluation: each new
            indexed-read bounds query is charged, and contractions retain their
            lexical ranges. Inactive branches are checked conservatively too.
            """
            if not node.has(sp.Indexed):
                return
            key = node, tuple((axis, bounds) for axis, bounds in ranges.items() if node.has(axis))
            if key in checked:
                return
            if isinstance(node, sp.Indexed):
                budget.spend(path="algebra.witness.read_bounds")
                indexed_coordinates(node, ranges)
            elif isinstance(node, (sp.Sum, sp.Product)):
                local = dict(ranges)
                for variable, start, stop in node.limits:
                    if not start.is_Integer or not stop.is_Integer or stop < start:
                        raise _Unproved
                    local[variable] = int(start), int(stop)
                read_bounds(node.function, local)
            else:
                for arg in node.args:
                    read_bounds(arg, ranges)
            checked.add(key)

        def visit(
            node: sp.Basic,
            ranges: dict[sp.Basic, tuple[int, int]],
            *,
            cache: dict[
                tuple[sp.Basic, tuple[tuple[sp.Basic, tuple[int, int]], ...]], sp.Basic
            ] = cache,
            pattern: str = pattern,
            domains_validated: list[bool] = domains_validated,
        ) -> sp.Basic:
            # Irrelevant enclosing axes cannot change a subtree's value or
            # indexed-read bounds. Keeping only syntactically occurring axes
            # reuses constants and independent reads across complete outputs.
            # A bound occurrence is retained conservatively as well.
            key = node, tuple((axis, bounds) for axis, bounds in ranges.items() if node.has(axis))
            if key in cache:
                return cache[key]
            if not isinstance(node, AppliedUndef):
                budget.spend(path="algebra.witness")
            if isinstance(node, AppliedUndef):
                if len(node.args) != 1:
                    raise _Unproved
                value = visit(node.args[0], ranges)
                name = node.func.__name__
                source_dtype, target_dtype = casts[name.removeprefix("defined_")]
                cast_key = value, source_dtype, target_dtype
                if cast_key not in checked_casts:
                    # Every source read is still specialized and bounds-checked
                    # above. The same exact dtype roundtrip need only run once.
                    budget.spend(path="algebra.witness.cast_validation")
                    valid = _representable_float(value, source_dtype)
                    if np.dtype(target_dtype).kind in "iu":
                        limits = np.iinfo(target_dtype)
                        valid = valid and bool(
                            value.is_Integer and limits.min <= value <= limits.max
                        )
                    else:
                        valid = (
                            valid
                            and dtypes.issubdtype(target_dtype, np.floating)
                            and _representable_float(value, target_dtype)
                        )
                    checked_casts[cast_key] = valid
                if not checked_casts[cast_key]:
                    raise _Unproved
                result = sp.S.One if name.startswith("defined_") else value
            elif isinstance(node, sp.Indexed):
                shape, indices = indexed_coordinates(node, ranges)
                if pattern == "one_hot" or (
                    pattern == "first_slice" and str(node.base.label) in floating_inputs
                ):
                    active = [i for i, extent in enumerate(shape) if extent > 1]
                    if pattern == "first_slice":
                        active = active[:1]
                    guard = sp.And(*(_coordinate_zero(indices[i], ranges, budget) for i in active))
                    result = (
                        guard
                        if str(node.base.label) in boolean_inputs
                        else sp.Piecewise((1, guard), (0, True))
                    )
                else:
                    result = (
                        sp.sympify(input_one(str(node.base.label)))
                        if str(node.base.label) in boolean_inputs
                        else sp.Integer(input_one(str(node.base.label)))
                    )
            elif isinstance(node, sp.Symbol) and str(node) in signatures:
                if signatures[str(node)]:
                    raise _Unproved
                result = (
                    sp.sympify(pattern == "one_hot" or input_one(str(node)))
                    if str(node) in boolean_inputs
                    else sp.Integer(pattern == "one_hot" or input_one(str(node)))
                )
            elif isinstance(node, sp.Mul) and finite_values and domains_validated[0]:
                # The caller certifies finite projected values under the
                # complete original domains. Only after every output's full
                # domain has passed for this input may a finite zero product
                # skip the remaining value computation. Domain evaluation
                # itself never takes this shortcut.
                args = []
                ordered = sorted(
                    node.args,
                    key=lambda arg: not isinstance(arg, (sp.Indexed, sp.Symbol, sp.Number)),
                )
                for index, arg in enumerate(ordered):
                    args.append(visit(arg, ranges))
                    if args[-1] == 0:
                        for skipped in ordered[index + 1 :]:
                            read_bounds(skipped, ranges)
                        break
                result = sp.Mul(*args)
            elif isinstance(node, sp.Piecewise):
                pairs = []
                for value, condition in node.args:
                    condition = visit(condition, ranges)
                    if condition == sp.false:
                        continue
                    value = visit(value, ranges)
                    if pairs and pairs[-1][0] == value:
                        condition = _boolean(sp.Or, (pairs.pop()[1], condition))
                    pairs.append((value, condition))
                    if condition == sp.true:
                        break
                result = sp.Piecewise(*pairs, evaluate=False) if pairs else sp.nan
            elif isinstance(node, Relational):
                # Subtracting the two sides can cancel even an out-of-bounds
                # input read. Only coordinate relations may bypass operand
                # evaluation; data relations must validate each read first.
                result = (
                    node
                    if input_address(node) or node.has(AppliedUndef)
                    else _bounded_truth(node, ranges)
                )
                if result == node:
                    result = node.func(*(visit(arg, ranges) for arg in node.args))
                if isinstance(result, Relational):
                    result = _bounded_truth(result, ranges)
            elif isinstance(node, (sp.And, sp.Or, sp.Not)):
                # Coverage domains may arrive in an intentionally unevaluated
                # form. Recheck their local Boolean identities after the exact
                # input specialization, even if coordinate atoms did not move.
                args = []
                decisive = sp.false if isinstance(node, sp.And) else sp.true
                for arg in node.args:
                    args.append(visit(arg, ranges))
                    if not isinstance(node, sp.Not) and args[-1] == decisive:
                        break
                # Negation normal form pushes Not through compound masks
                # without distributing conjunctions/disjunctions. It exposes
                # complementary literal pairs hidden inside page coverage.
                result = _boolean(node.func, args)
            elif isinstance(node, (sp.Sum, sp.Product)):
                local = dict(ranges)
                for variable, start, stop in node.limits:
                    if not start.is_Integer or not stop.is_Integer or stop < start:
                        raise _Unproved
                    local[variable] = int(start), int(stop)
                result = visit(constant_coordinates(node.function, local), local)
                for variable, start, stop in node.limits:
                    result = _constant_intervals(
                        result,
                        variable,
                        int(start),
                        int(stop),
                        budget,
                        product=isinstance(node, sp.Product),
                    )
            else:
                args = tuple(visit(arg, ranges) for arg in node.args)
                result = node if args == node.args else node.func(*args)
            cache[key] = result
            return result

        try:
            valid = True
            for condition in domains:
                # Complete output domains are identical across coordinate
                # searches. Their exact specialization depends on the input
                # pattern, source types and cast definitions, not the chosen
                # output coordinate. Cache only the final Boolean obligation;
                # value reads still undergo their own bounds/cast checks.
                key = (*domain_context, pattern, condition)
                if key not in domain_cache:
                    try:
                        domain_cache[key] = visit(condition, {}) == sp.true
                    except (_Unproved, sp.PolynomialError):
                        # False also means unproved: reject this witness
                        # pattern, without claiming the condition is false.
                        domain_cache[key] = False
                if not domain_cache[key]:
                    valid = False
                    break
            if not valid:
                continue
            domains_validated[0] = True
            a, b = visit(left, {}), visit(right, {})
            if any(
                value.free_symbols or value.has(sp.nan, sp.zoo, sp.oo, -sp.oo) for value in (a, b)
            ):
                continue
            decision = _exponential_equal(a, b, budget)
            if decision is False or decision is None and equal(a, b, budget) is False:
                return pattern, a, b
        except (_Unproved, sp.PolynomialError):
            continue
    return None
