"""Finite-domain bijections and exact witnesses, including practical Pallas sizes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jax.experimental import pallas as pl

from jax_reorder import Status, compare_functions
from jax_reorder.algebra.budget import Budget
from jax_reorder.algebra.sum_domains import compact_domain
from jax_reorder.algebra.witnesses import exact_witness


def grouped_sum(size, grouped, *, wrong=None, pallas=False, interpret=False):
    def calculate(x):
        value = jnp.sum(jnp.sum(x.reshape(2, size // 2), axis=1)) if grouped else jnp.sum(x)
        return value + 1 if wrong == "offset" else value * 2 if wrong == "weight" else value

    if not pallas:
        return calculate

    def kernel(x, out):
        out[:] = jnp.broadcast_to(calculate(x[:]), (size,))

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((size,), jnp.float32), interpret=interpret
    )


@pytest.mark.parametrize("pallas", [False, True])
@pytest.mark.parametrize("size", [1024, 4096, 2**20])
@pytest.mark.parametrize("wrong", [None, "offset", "weight"])
def test_grouped_sum_has_symbolic_positive_and_complete_negative_proofs(pallas, size, wrong):
    result = compare_functions(
        grouped_sum(size, False, pallas=pallas),
        grouped_sum(size, True, wrong=wrong, pallas=pallas),
        jax.ShapeDtypeStruct((size,), jnp.float32),
    )
    assert result.status is (Status.EQUIVALENT if wrong is None else Status.INEQUIVALENT)


def test_grouped_sum_independent_cpu_values():
    x = np.arange(1024, dtype=np.float32) % 17 - 8
    expected = np.sum(x)
    for wrong in (None, "offset", "weight"):
        output = grouped_sum(1024, True, wrong=wrong, pallas=True, interpret=True)(x)
        value = (
            expected + 1 if wrong == "offset" else expected * 2 if wrong == "weight" else expected
        )
        np.testing.assert_array_equal(output, np.full(1024, value, np.float32))


@pytest.mark.parametrize("start", [-2, 0, 3])
@pytest.mark.parametrize("form", ["offset", "stride", "separate", "mask", "reverse"])
def test_mixed_radix_inverse_preserves_every_coordinate_use(start, form):
    i, j = sp.symbols("i j", integer=True)
    x = sp.IndexedBase("x", shape=(100,), real=True)
    linear = 4 * i + j
    value = x[linear + 20]
    if form == "stride":
        value = x[2 * linear + 20]
    elif form == "separate":
        value *= 1 + i - j
    elif form == "mask":
        value *= sp.Piecewise((1, i > j), (0, True))
    elif form == "reverse":
        value += x[4 * i - j + 20]
    source = sp.Sum(value, (j, 0, 3), (i, start, start + 2))
    result = compact_domain(source, Budget(1000))
    assert len(result.limits) == 1
    assert sp.expand(source.doit() - result.doit()) == 0


def test_overlapping_nonbijective_coordinate_is_not_flattened():
    i, j = sp.symbols("i j", integer=True)
    x = sp.IndexedBase("x", shape=(16,), real=True)
    source = sp.Sum(x[3 * i + j], (j, 0, 3), (i, 0, 2))
    assert compact_domain(source, Budget(1000)) == source


@pytest.mark.parametrize("refusal", ["bounds", "domain", "opaque"])
def test_exact_witness_requires_original_domains_and_legal_input_reads(refusal):
    i = sp.Symbol("i", integer=True)
    x = sp.IndexedBase("x0", shape=(1024,), real=True)
    expression = sp.Sum(x[i], (i, 0, 1024 if refusal == "bounds" else 1023))
    domain = sp.false if refusal == "domain" else sp.true
    if refusal == "opaque":
        expression = sp.Function("cast_unknown")(expression)
    assert (
        exact_witness(
            expression,
            2 * expression,
            (domain,),
            [jax.ShapeDtypeStruct((1024,), jnp.float32)],
            Budget(1000),
        )
        is None
    )


@pytest.mark.parametrize("shape", [(), (64,)])
def test_exact_witness_preserves_boolean_input_values(shape):
    x = sp.Symbol("x0", boolean=True) if not shape else sp.IndexedBase("x0", shape=shape)[0]
    witness = exact_witness(
        sp.Not(x), x, (sp.true,), [jax.ShapeDtypeStruct(shape, jnp.bool_)], Budget(1000)
    )
    assert witness is not None
    assert set(witness[1:]) == {sp.true, sp.false}


@pytest.mark.parametrize(
    "value,source,target,accepted",
    [
        (127, "float32", "int8", True),
        (sp.Rational(3, 2), "float32", "int8", False),
        (128, "float32", "int8", False),
        (-129, "float32", "int8", False),
        (2049, "float16", "int16", False),
    ],
)
def test_integer_cast_witness_requires_representable_in_range_integer(
    value, source, target, accepted
):
    operation = sp.Function("known_cast", integer=True)
    witness = exact_witness(
        operation(sp.sympify(value)),
        sp.S.Zero,
        (sp.true,),
        [],
        Budget(1000),
        {"known_cast": (source, target)},
    )
    assert (witness is not None) is accepted


def test_lazy_universal_domains_preserve_each_finite_coordinate():
    from jax_reorder.algebra.expression import finite_product, finite_sum, indicator, substitute

    i, k, x = sp.symbols("i k x", integer=True)
    nested = finite_sum(sp.Piecewise((x, k < 2), (1, True)), (k, 0, 2)) + i
    domain = finite_product(indicator(sp.Gt(nested, 0, evaluate=False)), (i, 0, 3))
    renamed = substitute(domain, {x: x + 1})
    for value in (-3, -1, 0, 2):
        expected = all(
            sum(value + 1 if column < 2 else 1 for column in range(3)) + row > 0 for row in range(4)
        )
        condition = sp.Eq(substitute(renamed, {x: sp.Integer(value)}), 1, evaluate=False)
        witness = exact_witness(sp.S.One, sp.S.Zero, (condition,), [], Budget(1000))
        assert (witness is not None) is expected


def test_flat_extrema_keeps_shared_composite_operands():
    from jax_reorder.algebra.extrema import FlatMax, FlatMin

    i = sp.Symbol("i", integer=True)
    x = sp.IndexedBase("x", shape=(128,))
    shared = sp.Sum(x[i], (i, 0, 127))
    operands = [shared + n for n in range(40)]
    for operation in (FlatMax, FlatMin):
        value = operation(*operands, *reversed(operands))
        assert set(value.args) == set(operands)
        assert value.xreplace({shared: sp.Integer(7)}) == (46 if operation is FlatMax else 7)


def test_alpha_identity_retains_captured_axes_and_limit_order():
    from jax_reorder.algebra.expression import finite_sum, same_bound_expression

    i, j, k = sp.symbols("i j k", integer=True)
    x = sp.IndexedBase("x", shape=(8, 8))
    left = finite_sum(x[i, k], (i, 0, 7))
    assert same_bound_expression(left, finite_sum(x[j, k], (j, 0, 7)))
    assert not same_bound_expression(left, finite_sum(x[j, i], (j, 0, 7)))
    assert not same_bound_expression(left, finite_sum(x[j, k], (j, 1, 7)))


@pytest.mark.parametrize("offset,valid", [(0, True), (16, False), (-1, False)])
def test_exact_witness_preserves_expanded_remainder_bounds(offset, valid):
    from jax_reorder.algebra.expression import finite_sum

    k = sp.Symbol("k", integer=True)
    x = sp.IndexedBase("x0", shape=(16,))
    expression = finite_sum(x[k - 16 * sp.floor(k / 16) + offset], (k, 0, 127))
    witness = exact_witness(
        expression,
        sp.S.Zero,
        (),
        [jax.ShapeDtypeStruct((16,), jnp.float32)],
        Budget(1000),
    )
    assert (witness is not None) is valid


def test_coordinate_read_renames_a_captured_sum_binder():
    from jax_reorder.algebra.expression import SymbolicTensor, coordinates, finite_sum

    coordinate = coordinates((8,))[0]
    bound = sp.Symbol("bound", integer=True)
    expression = finite_sum(bound + coordinate, (bound, 0, 2))
    tensor = SymbolicTensor((8,), jnp.float32, expression)
    concrete, _ = tensor.read((bound,))
    assert concrete.doit() == 3 * bound + 3
    assert tensor.read((coordinate,))[0] is expression


def test_lazy_sum_preserves_piecewise_and_nested_domains():
    from jax_reorder.algebra.expression import finite_sum, substitute

    i, j, free = sp.symbols("i j free", integer=True)
    function = sp.Piecewise((i + free, i < j), (0, True))
    lazy = finite_sum(finite_sum(function, (i, 0, 3)), (j, 0, 4))
    regular = sp.Sum(function, (i, 0, 3), (j, 0, 4))
    assert lazy.doit() == regular.doit()
    assert substitute(lazy, {free: sp.Integer(2)}).doit() == regular.subs(free, 2).doit()


@pytest.mark.parametrize(
    "left,right,answer",
    [
        (sp.E, sp.exp(sp.Rational(1, 2)) ** 2, True),
        (sp.exp(sp.Rational(-4000001, 4)), sp.exp(sp.Rational(-4000003, 4)), False),
        (
            (1 + sp.exp(sp.Rational(1, 3))) ** 2,
            1 + 2 * sp.exp(sp.Rational(1, 3)) + sp.exp(sp.Rational(2, 3)),
            True,
        ),
        (1 / (1 + sp.exp(-1000000)), 1 / (1 + sp.exp(-1000001)), False),
        (sp.exp(sp.Symbol("q", rational=True)), 1, None),
        (sp.exp(sp.sqrt(2)), 1, None),
        (sp.Pow(sp.S.Zero, -1, evaluate=False), 1, None),
        (sp.Pow(sp.Add(sp.E, -sp.E, evaluate=False), -1, evaluate=False), 1, None),
        (sp.Function("opaque")(0), 1, None),
    ],
)
def test_exact_closed_exponential_ring(left, right, answer):
    from jax_reorder.algebra.witnesses import _exponential_equal

    assert _exponential_equal(sp.sympify(left), sp.sympify(right), Budget(1000)) is answer


@pytest.mark.parametrize("cast_in_value", [False, True])
@pytest.mark.parametrize(
    "value,source,target,accepted",
    [
        (sp.Rational(3, 2), "float32", "float16", True),
        (sp.Rational(1, 3), "float32", "float16", False),
        (2049, "float32", "float16", False),
        (65536, "float32", "float16", False),
        (sp.Rational(1, 2**24), "float32", "float16", True),
        (sp.Rational(1, 2**25), "float32", "float16", False),
        (257, "float32", "bfloat16", False),
        (256, "float32", "bfloat16", True),
        (16777217, "float32", "float64", False),
        (sp.oo, "float32", "float16", False),
        (-sp.oo, "float32", "float16", False),
        (sp.nan, "float32", "float16", False),
    ],
)
def test_float_cast_witness_requires_exact_finite_values_and_domains(
    value, source, target, accepted, cast_in_value
):
    operation = sp.Function("known_cast", real=True)
    defined = sp.Function("defined_known_cast")
    # An invalid cast in another output's domain must also block a witness
    # whose differing value itself does not use that cast.
    witness = exact_witness(
        operation(sp.sympify(value)) if cast_in_value else sp.S.One,
        sp.S.Zero,
        (sp.Eq(defined(sp.sympify(value)), 1),),
        [],
        Budget(1000),
        {"known_cast": (source, target)},
    )
    assert (witness is not None) is accepted


def test_witness_boolean_complements_preserve_noncomplementary_guards():
    from jax_reorder.algebra.witnesses import _boolean

    a, b = sp.symbols("a b", boolean=True)
    for operation, dual, expected in ((sp.Or, sp.And, sp.true), (sp.And, sp.Or, sp.false)):
        assert _boolean(operation, (dual(a, b), sp.Not(a), sp.Not(b))) == expected
        incomplete = _boolean(operation, (dual(a, b), sp.Not(a)))
        assert incomplete.subs({a: True, b: True}) == (operation == sp.Or)
        assert incomplete.subs({a: True, b: False}) == sp.false


@pytest.mark.parametrize("coefficient", [-17, -16, -15, 0, 16])
@pytest.mark.parametrize("offset", [-5, 0, 9])
def test_remainder_bounds_keep_unmatched_coefficients_and_offsets(coefficient, offset):
    from jax_reorder.algebra.witnesses import _bounds

    i = sp.Symbol("i", integer=True)
    expression = i + coefficient * sp.floor(i / 16) + offset
    lo, hi = _bounds(expression, {i: (-20, 40)})
    values = [int(expression.subs(i, value)) for value in range(-20, 41)]
    assert lo <= min(values) <= max(values) <= hi


@pytest.mark.parametrize("denominator", [-16, 0, sp.Rational(3, 2)])
def test_remainder_recovery_does_not_assume_arbitrary_positive_integer_divisors(denominator):
    from jax_reorder.algebra.witnesses import _bounds, _Unproved

    real = sp.Symbol("real", real=True)
    expression = real - denominator * sp.floor(real / denominator)
    with pytest.raises(_Unproved):
        _bounds(expression, {})


@pytest.mark.parametrize("offset,allowed", [(0, True), (65536, False)])
def test_isolated_input_witness_checks_cast_domains_before_finite_zero_products(offset, allowed):
    x, y, z = sp.symbols("x0 x1 x2", real=True)
    cast = sp.Function("finite_cast", real=True)
    defined = sp.Function("defined_finite_cast")
    value = sp.Add(sp.Mul(x, cast(y + offset), evaluate=False), z, evaluate=False)
    witness = exact_witness(
        value,
        2 * value,
        (sp.Eq(defined(y + offset), 1),),
        [jax.ShapeDtypeStruct((), jnp.float32)] * 3,
        Budget(1000),
        {"finite_cast": ("float32", "float16")},
        finite_values=True,
    )
    assert (witness is not None) is allowed
    if allowed:
        assert witness == ("only_input_2", 1, 2)


def test_isolated_zero_product_cannot_skip_another_outputs_log_domain():
    x, y, z = sp.symbols("x0 x1 x2", real=True)
    cast = sp.Function("finite_cast", real=True)
    value = sp.Add(sp.Mul(x, cast(y), evaluate=False), z, evaluate=False)
    witness = exact_witness(
        value,
        2 * value,
        (x > 0,),
        [jax.ShapeDtypeStruct((), jnp.float32)] * 3,
        Budget(1000),
        {"finite_cast": ("float32", "float16")},
        finite_values=True,
    )
    assert witness == ("one", 2, 4)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("valid", [False, True])
def test_finite_zero_product_still_checks_skipped_read_bounds(nested, valid):
    from jax_reorder.algebra.expression import finite_sum

    x = sp.IndexedBase("x0", shape=(1,))
    i = sp.Symbol("i", integer=True)
    skipped = finite_sum(x[i], (i, 0, int(not valid))) if nested else x[int(not valid)]
    value = sp.Mul(x[0], skipped, evaluate=False)
    witness = exact_witness(
        value,
        sp.Integer(2),
        (),
        [jax.ShapeDtypeStruct((1,), jnp.float32)],
        Budget(1000),
        finite_values=True,
    )
    assert (witness is not None) is valid


@pytest.mark.parametrize("page_slot,offset", [(1, 0), (2, 0), (1, 2)])
def test_exact_witness_specializes_nested_page_reads_before_payload_selection(page_slot, offset):
    pages = sp.IndexedBase("x0", shape=(2,), integer=True)
    data = sp.IndexedBase("x1", shape=(2,), real=True)
    page = sp.Mod(pages[page_slot], 2)
    witness = exact_witness(
        data[page + offset],
        data[1 - page + offset],
        (),
        [jax.ShapeDtypeStruct((2,), jnp.int32), jax.ShapeDtypeStruct((2,), jnp.float32)],
        Budget(1000),
    )
    assert witness == (("one_hot", 1, 0) if page_slot == 1 and offset == 0 else None)


def test_exact_witness_keeps_page_reads_correlated_inside_a_contraction():
    from jax_reorder.algebra.expression import finite_sum

    pages = sp.IndexedBase("x0", shape=(3,), integer=True)
    data = sp.IndexedBase("x1", shape=(2,), real=True)
    i = sp.Symbol("page", integer=True)
    witness = exact_witness(
        finite_sum(data[pages[i]], (i, 0, 2)),
        finite_sum(data[1 - pages[i]], (i, 0, 2)),
        (),
        [jax.ShapeDtypeStruct((3,), jnp.int32), jax.ShapeDtypeStruct((2,), jnp.float32)],
        Budget(1000),
    )
    assert witness == ("one_hot", 2, 1)


@pytest.mark.parametrize("page_slot,offset", [(0, 0), (1, 0), (0, 2)])
def test_finite_zero_product_checks_nested_address_reads_when_skipping_payload(page_slot, offset):
    zero = sp.Symbol("x0", real=True)
    pages = sp.IndexedBase("x1", shape=(1,), integer=True)
    data = sp.IndexedBase("x2", shape=(2,), real=True)
    left = sp.Mul(zero, data[pages[page_slot] + offset], evaluate=False)
    witness = exact_witness(
        left,
        sp.S.One,
        (sp.Eq(zero, 0),),
        [
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((2,), jnp.float32),
        ],
        Budget(1000),
        finite_values=True,
    )
    assert witness == (("zero", 0, 1) if page_slot == 0 and offset == 0 else None)


@pytest.mark.parametrize("invalid", [None, "inner_read", "outer_read", "other_output_cast"])
@pytest.mark.parametrize("cast_in_value", [False, True])
def test_first_slice_witness_keeps_metadata_and_complete_domains(invalid, cast_in_value):
    from jax_reorder.algebra.expression import finite_sum

    q = sp.IndexedBase("x0", shape=(1, 2, 2), real=True)
    k = sp.IndexedBase("x1", shape=(1, 2, 2, 2), real=True)
    # An additional floating input unused by these scalar outputs must not
    # put an ineffective isolated-input candidate ahead of first_slice.
    pages = sp.IndexedBase("x3", shape=(2,), integer=True)
    cast = sp.Function("exact_cast", real=True)
    lane = sp.Symbol("lane", integer=True)
    page = sp.Mod(pages[2 if invalid == "inner_read" else 0], 2)
    if invalid == "outer_read":
        page += 2

    def score(page, token):
        value = k[0, page, token, lane]
        return finite_sum(q[0, 0, lane] * (cast(value) if cast_in_value else value), (lane, 0, 1))

    original = score(page, 0)
    rewritten = score(sp.Mod(page + 1, 2), 0)
    defined = sp.Function("defined_exact_cast")
    # This additional output would require nonrepresentable exp(-1) for
    # one-hot lanes. A whole first slice keeps both token scores identical.
    domains = [sp.Eq(defined(sp.exp(score(page, 1) - original)), 1)]
    if invalid == "other_output_cast":
        domains.append(sp.Eq(defined(sp.Rational(1, 3)), 1))
    witness = exact_witness(
        original,
        rewritten,
        domains,
        [
            jax.ShapeDtypeStruct((1, 2, 2), jnp.float32),
            jax.ShapeDtypeStruct((1, 2, 2, 2), jnp.float32),
            jax.ShapeDtypeStruct((1, 2, 2, 2), jnp.float32),
            jax.ShapeDtypeStruct((2,), jnp.int32),
        ],
        Budget(3000),
        {"exact_cast": ("float32", "bfloat16")},
    )
    assert witness == (("first_slice", 2, 0) if invalid is None else None)


def test_witness_coordinate_quotients_preserve_shadowed_binders_and_ranges():
    from jax_reorder.algebra.expression import finite_sum

    i = sp.Symbol("i", integer=True)
    quotient = sp.floor(i / 4)
    inner = finite_sum(quotient, (i, 8, 11))
    nested = finite_sum(quotient + inner, (i, 0, 3))
    assert exact_witness(nested, sp.S.Zero, (), [], Budget(1000)) == ("one", 32, 0)
    # The same syntactic quotient in another scope has a different value.
    assert exact_witness(finite_sum(quotient, (i, 0, 3)), inner, (), [], Budget(1000)) == (
        "one",
        0,
        8,
    )


@pytest.mark.parametrize("active_oob", [False, True])
def test_witness_constant_coordinate_guards_keep_reachable_read_bounds(active_oob):
    from jax_reorder.algebra.expression import finite_sum

    i = sp.Symbol("i", integer=True)
    data = sp.IndexedBase("x0", shape=(1,), real=True)
    body = sp.Piecewise((data[int(active_oob)], sp.Eq(sp.floor(i / 4), 0)), (data[1], True))
    witness = exact_witness(
        finite_sum(body, (i, 0, 3)),
        sp.S.Zero,
        (),
        [jax.ShapeDtypeStruct((1,), jnp.float32)],
        Budget(1000),
    )
    assert witness == (None if active_oob else ("one", 4, 0))


def test_witness_coordinate_quotients_never_bound_input_values_by_shape():
    from jax_reorder.algebra.expression import finite_sum

    i = sp.Symbol("i", integer=True)
    metadata = sp.IndexedBase("x0", shape=(1,), integer=True)
    body = sp.Piecewise((2, sp.Eq(sp.floor(3 * metadata[0] / 2), 0)), (3, True))
    assert exact_witness(
        finite_sum(body, (i, 0, 1)),
        sp.Integer(4),
        (),
        [jax.ShapeDtypeStruct((1,), jnp.int32)],
        Budget(1000),
    ) == ("one", 6, 4)


@pytest.mark.parametrize(
    "first,second",
    [
        ((257, "float32", "float16"), (257, "float32", "bfloat16")),
        ((2049, "float32", "float32"), (2049, "float16", "float32")),
        ((256, "float32", "bfloat16"), (257, "float32", "bfloat16")),
    ],
)
def test_witness_cast_cache_separates_exact_values_and_both_dtypes(first, second):
    a, b = sp.Function("defined_cast_a"), sp.Function("defined_cast_b")
    assert (
        exact_witness(
            sp.S.One,
            sp.S.Zero,
            (sp.Eq(a(first[0]), 1), sp.Eq(b(second[0]), 1)),
            [],
            Budget(1000),
            {"cast_a": first[1:], "cast_b": second[1:]},
        )
        is None
    )


@pytest.mark.parametrize("finite_values", [False, True])
@pytest.mark.parametrize("consumer", ["product", "sin", "relation"])
def test_coordinate_folding_cannot_erase_unvisited_out_of_bounds_reads(consumer, finite_values):
    from jax_reorder.algebra.expression import finite_sum

    i = sp.Symbol("i", integer=True)
    data = sp.IndexedBase("x0", shape=(1,), real=True)
    product = sp.Mul(sp.floor(i / 4), data[1], evaluate=False)
    body = (
        sp.sin(product, evaluate=False)
        if consumer == "sin"
        else sp.Eq(product, 0, evaluate=False)
        if consumer == "relation"
        else product
    )
    if consumer == "relation":
        left, domains = sp.S.Zero, (sp.Eq(sp.Product(body, (i, 0, 3)), 1),)
    else:
        left, domains = finite_sum(body, (i, 0, 3)), ()
    assert (
        exact_witness(
            left,
            sp.S.One,
            domains,
            [jax.ShapeDtypeStruct((1,), jnp.float32)],
            Budget(1000),
            finite_values=finite_values,
        )
        is None
    )


@pytest.mark.parametrize("as_domain", [False, True])
def test_witness_equal_read_relations_check_both_original_read_bounds(as_domain):
    data = sp.IndexedBase("x0", shape=(1,), real=True)
    condition = sp.Eq(data[1], data[1], evaluate=False)
    left = sp.S.One if as_domain else sp.Piecewise((1, condition), (0, True), evaluate=False)
    assert (
        exact_witness(
            left,
            sp.S.Zero,
            (condition,) if as_domain else (),
            [jax.ShapeDtypeStruct((1,), jnp.float32)],
            Budget(1000),
        )
        is None
    )


@pytest.mark.parametrize("name", ["cast", "defined_cast"])
def test_equal_closed_cast_relations_still_validate_exact_representation(name):
    operation = sp.Function(name)
    invalid = operation(sp.Rational(1, 3))
    assert (
        exact_witness(
            sp.S.One,
            sp.S.Zero,
            (sp.Eq(invalid, invalid, evaluate=False),),
            [],
            Budget(1000),
            {"cast": ("float32", "float16")},
        )
        is None
    )


def test_witness_coordinate_cache_keeps_independent_axes_with_equal_ranges_distinct():
    from jax_reorder.algebra.expression import finite_sum

    i, j = sp.symbols("i j", integer=True)
    dependent = sp.floor(
        sp.Mul(sp.Rational(1, 2), sp.Add(i, -i, 1, evaluate=False), evaluate=False),
        evaluate=False,
    )
    independent = sp.floor((i - j + 1) / 2)
    left = sp.Add(
        finite_sum(dependent, (i, 0, 1)),
        finite_sum(independent, (i, 0, 1), (j, 0, 1)),
        evaluate=False,
    )
    # The first quotient is constant. The second is not, despite equal axis
    # ranges; this witness evaluator conservatively leaves that sum unproved.
    assert exact_witness(left, sp.Integer(2), (), [], Budget(1000)) is None


@pytest.mark.parametrize("operation", [sp.floor, sp.Mod])
def test_witness_coordinate_bounds_cannot_cancel_closed_invalid_casts(operation):
    from jax_reorder.algebra.expression import finite_sum

    i = sp.Symbol("i", integer=True)
    cast = sp.Function("cast")
    invalid = cast(sp.Rational(1, 3))
    numerator = sp.Add(invalid, -invalid, 1, evaluate=False)
    quotient = (
        sp.floor(sp.Mul(sp.Rational(1, 4), numerator, evaluate=False), evaluate=False)
        if operation == sp.floor
        else sp.Mod(numerator, 4, evaluate=False)
    )
    assert (
        exact_witness(
            finite_sum(quotient, (i, 0, 3)),
            sp.Integer(7),
            (),
            [],
            Budget(1000),
            {"cast": ("float32", "float16")},
        )
        is None
    )
