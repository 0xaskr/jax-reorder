"""Public Pallas algebra bridge: full outputs, CAS evidence and refusal paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax._src.state import types as st
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
    compare_pallas_kernels,
)
from tests.test_algebra_scalar_ir import BOOLEAN_SCALARS, boolean_scalar
from tests.test_pallas_semantic_slice import build, trace
from tests.test_pallas_tensor_memory import rank_one_broadcast, rank_one_reduction


def pointwise(fn, block=128, *, size=1024, inputs=1, outputs=1, dtype=jnp.float32, interpret=False):
    sig = jax.ShapeDtypeStruct((size,), dtype)
    spec = pl.BlockSpec((block,), lambda i: (i,))

    def kernel(*refs):
        values = fn(*(x[:] for x in refs[:inputs]))
        values = (values,) if outputs == 1 else values
        for ref, value in zip(refs[inputs:], values, strict=True):
            ref[:] = jnp.broadcast_to(value, (block,))

    return pl.pallas_call(
        kernel,
        out_shape=sig if outputs == 1 else (sig,) * outputs,
        grid=(size // block,),
        in_specs=(spec,) * inputs,
        out_specs=spec if outputs == 1 else (spec,) * outputs,
        interpret=interpret,
    )


def traced(fn, size=1024, inputs=1):
    return jax.make_jaxpr(fn)(*(jax.ShapeDtypeStruct((size,), jnp.float32),) * inputs)


def test_shared_address_coordinates_keep_source_ports_and_output_offsets():
    import sympy as sp

    from jax_reorder.algebra.adapter.pallas import project
    from jax_reorder.algebra.budget import Budget
    from jax_reorder.algebra.symbolic import SymbolicBuilder

    def kernel(a, b, first, repeated, shifted):
        first[...] = a[:, :128] + b[:, :128]
        repeated[...] = a[:, :128] + 2 * b[:, :128]
        shifted[...] = a[:, 128:] + b[:, 128:]

    source = jax.make_jaxpr(
        pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((8, 128), jnp.float32),) * 3,
            compiler_params=tpu.CompilerParams(),
        )
    )(*(jax.ShapeDtypeStruct((8, 256), jnp.float32),) * 2)
    graph, _ = project(source.jaxpr, Budget(10000))
    values = SymbolicBuilder(10000).graph(graph)
    a = np.arange(8 * 256, dtype=np.float32).reshape((8, 256))
    b = 3 * a + 7
    references = (a[:, :128] + b[:, :128], a[:, :128] + 2 * b[:, :128], a[:, 128:] + b[:, 128:])
    inputs = {"x0": a, "x1": b}
    for value, reference in zip(values, references, strict=True):
        assert value.defined == value.memory_validity == sp.true
        for point in np.ndindex(reference.shape):
            expression, _ = value.read(point)
            actual = expression.xreplace(
                {
                    read: float(inputs[str(read.base.label)][tuple(map(int, read.indices))])
                    for read in expression.atoms(sp.Indexed)
                }
            )
            assert float(actual) == reference[point]


def boolean_dot_pallas(mode="dot", *, different=False, preferred=None, interpret=False):
    call = pointwise(lambda x: x, size=128, interpret=interpret)

    def outer(x):
        left, right = x > 0, x < 2
        if mode == "dot":
            value = jax.lax.dot_general(
                left, right, (((0,), (0,)), ((), ())), preferred_element_type=preferred
            )
        elif mode == "count":
            value = jnp.dot(left.astype(jnp.float32), right.astype(jnp.float32))
        else:
            value = jnp.max(left & right)
        output = value.astype(jnp.float32)
        return call(jnp.broadcast_to(output + 1 if different else output, (128,)))

    return outer


@pytest.mark.parametrize("entry", ["algebra", "functions"])
@pytest.mark.parametrize("preferred", [None, jnp.float32], ids=["boolean", "count"])
def test_boolean_dot_composes_with_pallas(entry, preferred):
    comparisons = [(False, Status.EQUIVALENT)]
    if preferred is None:
        comparisons.append((True, Status.INEQUIVALENT))
    for different, status in comparisons:
        functions = (
            boolean_dot_pallas(preferred=preferred),
            boolean_dot_pallas("max" if preferred is None else "count", different=different),
        )
        result = (
            compare_algebraic_jaxprs(*(traced(fn, size=128) for fn in functions))
            if entry == "algebra"
            else compare_functions(*functions, jax.ShapeDtypeStruct((128,), jnp.float32))
        )
        assert result.status is status
    for scalar in (-1, 1, 3):
        inputs = np.full(128, scalar, np.float32)
        matches = (inputs > 0) & (inputs < 2)
        expected = np.any(matches) if preferred is None else np.sum(matches)
        np.testing.assert_array_equal(
            boolean_dot_pallas(preferred=preferred, interpret=True)(inputs),
            np.full(128, expected, np.float32),
        )


@pytest.mark.parametrize("entry", ["algebra", "functions"])
def test_boolean_dot_numeric_accumulator_is_not_boolean_or_in_pallas(entry):
    def build(preferred, *, interpret=False):
        call = boolean_dot_pallas(preferred=preferred, interpret=interpret)
        return lambda x: call(jnp.ones_like(x))

    functions = (build(jnp.float32), build(None))
    result = (
        compare_algebraic_jaxprs(*(traced(fn, size=128) for fn in functions))
        if entry == "algebra"
        else compare_functions(*functions, jax.ShapeDtypeStruct((128,), jnp.float32))
    )
    assert result.status is Status.INEQUIVALENT
    inputs = np.zeros(128, np.float32)
    for preferred, expected in ((jnp.float32, 128), (None, 1)):
        np.testing.assert_array_equal(
            build(preferred, interpret=True)(inputs), np.full(128, expected, np.float32)
        )


@pytest.mark.parametrize("obligation", ["defined", "memory"])
def test_boolean_dot_keeps_full_operand_obligations(obligation):
    import sympy as sp

    from jax_reorder.algebra.expression import SymbolicTensor, coordinates
    from jax_reorder.algebra.pallas import _operation
    from jax_reorder.algebra.symbolic import SymbolicBuilder

    shape = (4,)
    signature = jax.ShapeDtypeStruct(shape, jnp.bool_)
    equation = jax.make_jaxpr(jnp.dot)(signature, signature).jaxpr.eqns[0]
    (index,) = coordinates(shape)
    cells = sp.IndexedBase("cells", shape=shape, real=True)
    domain = cells[index] > 0 if obligation == "defined" else sp.true
    validity = index < 3 if obligation == "memory" else sp.true
    values = [
        SymbolicTensor(shape, np.dtype("bool"), sp.false),
        SymbolicTensor(shape, np.dtype("bool"), sp.true, domain, memory_validity=validity),
    ]
    result = _operation(SymbolicBuilder(1000), equation, values)
    assert result.expression == sp.false
    assert result.defined == (
        sp.And(*(cells[i] > 0 for i in range(4))) if obligation == "defined" else sp.true
    )
    assert result.memory_validity == (sp.false if obligation == "memory" else sp.true)


@pytest.mark.parametrize("entry", ["algebra", "functions"])
@pytest.mark.parametrize(
    ("name", "placement"),
    [(name, "outer") for name in BOOLEAN_SCALARS]
    + [(name, "direct") for name in ("lt", "le", "gt", "ge")],
)
def test_boolean_scalar_casts_compose_with_pallas(name, placement, entry):
    def build(reference=False, *, different=False, interpret=False):
        def operation(x):
            value = boolean_scalar(name, x > 0, x < 2, reference=reference)
            result = value.astype(jnp.float32)
            return result + 1 if different else result

        call = pointwise(
            (lambda x: x) if placement == "outer" else operation,
            size=128,
            interpret=interpret,
        )
        return (lambda x: call(operation(x))) if placement == "outer" else call

    for different, status in ((False, Status.EQUIVALENT), (True, Status.INEQUIVALENT)):
        left, right = build(), build(True, different=different)
        result = (
            compare_algebraic_jaxprs(*(traced(fn, size=128) for fn in (left, right)))
            if entry == "algebra"
            else compare_functions(left, right, jax.ShapeDtypeStruct((128,), jnp.float32))
        )
        assert result.status is status
        inputs = np.linspace(-3, 3, 128, dtype=np.float32)
        original = np.asarray(build(interpret=True)(inputs))
        candidate = np.asarray(build(True, different=different, interpret=True)(inputs))
        np.testing.assert_array_equal(candidate, 1 + original if different else original)


@pytest.mark.parametrize("entry", ["algebra", "functions"])
@pytest.mark.parametrize("threshold", [0, 64])
def test_boolean_reduce_max_can_feed_pallas(entry, threshold):
    def build(literal=None, *, interpret=False):
        call = pointwise(lambda x: x, size=128, interpret=interpret)

        def outer(x):
            value = (
                jnp.max(jnp.arange(128, dtype=jnp.int32) < threshold).astype(jnp.float32)
                if literal is None
                else jnp.float32(literal)
            )
            return call(jnp.broadcast_to(value, (128,)))

        return outer

    expected = np.float32(np.max(np.arange(128) < threshold))
    for literal, status in ((expected, Status.EQUIVALENT), (1 - expected, Status.INEQUIVALENT)):
        functions = (build(), build(literal))
        result = (
            compare_algebraic_jaxprs(*(traced(fn, size=128) for fn in functions))
            if entry == "algebra"
            else compare_functions(*functions, jax.ShapeDtypeStruct((128,), jnp.float32))
        )
        assert result.status is status
    inputs = np.zeros(128, np.float32)
    np.testing.assert_array_equal(build(interpret=True)(inputs), np.full(128, expected))


@pytest.mark.parametrize("name", ["lt", "le", "gt", "ge"])
def test_boolean_where_can_feed_order_comparisons(name):
    expected = int(name in ("le", "ge"))

    def build(literal=None, *, interpret=False):
        def operation(x):
            if literal is not None:
                return jnp.full_like(x, literal)
            predicate = jnp.where(jnp.arange(128) < 64, x > 0, False)
            zero = jnp.zeros_like(predicate)
            left, right = (zero, predicate) if name in ("le", "gt") else (predicate, zero)
            return getattr(jax.lax, name)(left, right).astype(jnp.float32)

        return pointwise(operation, size=128, interpret=interpret)

    for literal, status in ((expected, Status.EQUIVALENT), (1 - expected, Status.INEQUIVALENT)):
        result = compare_algebraic_jaxprs(
            *(traced(fn, size=128) for fn in (build(), build(literal)))
        )
        assert result.status is status
    inputs = np.linspace(-3, 3, 128, dtype=np.float32)
    np.testing.assert_array_equal(build(interpret=True)(inputs), np.full(128, expected))


@pytest.mark.parametrize("size", [128, 1024])
def test_boolean_constant_branches_do_not_create_deep_expressions(size):
    def build(literal=None, *, interpret=False):
        call = pointwise(lambda x: x, size=size, interpret=interpret)

        def outer(x):
            values = jnp.asarray([False, True] * (size // 2), jnp.bool_)
            bias = (
                jax.lax.le(jnp.zeros_like(values), values).astype(jnp.float32)
                if literal is None
                else jnp.float32(literal)
            )
            return call(x + bias)

        return outer

    for literal, status in ((1, Status.EQUIVALENT), (0, Status.INEQUIVALENT)):
        result = compare_algebraic_jaxprs(
            *(traced(fn, size=size) for fn in (build(), build(literal)))
        )
        assert result.status is status
    inputs = np.zeros(size, np.float32)
    np.testing.assert_array_equal(build(interpret=True)(inputs), np.ones(size, np.float32))


def coordinate_cast(
    before, *, placement="direct", dtype=jnp.float32, base=0, offset=1, interpret=False
):
    size = 256
    block = 256 if jnp.dtype(dtype).itemsize == 2 else 128

    def bias(extent, start):
        coordinates = jnp.arange(extent, dtype=jnp.int32) + start + base
        return (
            (coordinates + offset).astype(dtype)
            if before
            else coordinates.astype(dtype) + jnp.asarray(offset, dtype)
        )

    def operation(x):
        start = block * pl.program_id(0)

        def body():
            return (x + bias(block, start)).astype(dtype)

        return jax.lax.cond(pl.program_id(0) >= 0, body, body) if placement == "cond" else body()

    call = pointwise(
        (lambda x: x) if placement == "outer" else operation,
        block,
        size=size,
        dtype=dtype,
        interpret=interpret,
    )
    return (lambda x: call(x + bias(size, 0))) if placement == "outer" else call


def integer_constant_cast(number, source_dtype, target_dtype, *, literal=None, interpret=False):
    identity = pointwise(lambda x: x, size=128, interpret=interpret)

    def outer(x):
        value = (
            (
                jnp.arange(128, dtype=source_dtype) * 0 + jnp.asarray(number, dtype=source_dtype)
            ).astype(target_dtype)
            if literal is None
            else jnp.full((128,), literal, target_dtype)
        )
        return identity(x + value.astype(jnp.float32))

    return outer


def outer_iota_cast(dtype, *, size=512, cast_before_call=False, offset=0, interpret=False):
    call = pointwise(lambda x: x.astype(jnp.float32), block=size, size=size, interpret=interpret)

    def outer(x):
        values = jnp.arange(size, dtype=dtype)
        if offset:
            values = values + offset
        if cast_before_call:
            values = values.astype(jnp.float32)
        return call(values)

    return outer


@pytest.mark.parametrize("entry", ["algebra", "functions", "local"])
@pytest.mark.parametrize("cast_before_call", [False, True])
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.float32, jnp.float64])
def test_outer_float_iota_cannot_be_used_as_integer_coordinates(entry, cast_before_call, dtype):
    with jax.enable_x64():
        functions = tuple(
            outer_iota_cast(source, cast_before_call=cast_before_call)
            for source in (dtype, jnp.int32)
        )
        sources = tuple(traced(fn, size=512) for fn in functions)
        assert compare_jaxprs(sources[0], sources[0]).status is Status.EQUIVALENT
        with pytest.raises(UnsupportedJaxprError, match="integer.*dtype"):
            if entry == "algebra":
                compare_algebraic_jaxprs(*sources)
            elif entry == "functions":
                compare_functions(*functions, jax.ShapeDtypeStruct((512,), jnp.float32))
            else:
                compare_pallas_kernels(*sources, check_protocol=True)


@pytest.mark.parametrize(
    ("dtype", "size", "index"), [(jnp.bfloat16, 512, 257), (jnp.float16, 4096, 2049)]
)
def test_outer_float_iota_rounding_survives_the_kernel_boundary(dtype, size, index):
    arguments = np.zeros(size, np.float32)
    reference = np.arange(size, dtype=np.int32).astype(dtype).astype(np.float32)
    expected_integer = np.arange(size, dtype=np.float32)
    assert reference[index] == index - 1
    for source, expected in ((dtype, reference), (jnp.int32, expected_integer)):
        function = outer_iota_cast(source, size=size, interpret=True)
        for execute in (function, jax.jit(function)):
            np.testing.assert_array_equal(execute(arguments), expected)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.uint32])
@pytest.mark.parametrize("offset", [0, 1])
def test_outer_integer_iota_keeps_exact_cast_comparisons(dtype, offset):
    functions = (
        outer_iota_cast(dtype),
        outer_iota_cast(dtype, cast_before_call=True, offset=offset),
    )
    expected = Status.EQUIVALENT if offset == 0 else Status.INEQUIVALENT
    assert compare_algebraic_jaxprs(*(traced(fn, size=512) for fn in functions)).status is expected
    assert (
        compare_functions(*functions, jax.ShapeDtypeStruct((512,), jnp.float32)).status is expected
    )
    arguments = np.zeros(512, np.float32)
    for before, delta in ((False, 0), (True, offset)):
        actual = outer_iota_cast(dtype, cast_before_call=before, offset=delta, interpret=True)(
            arguments
        )
        np.testing.assert_array_equal(actual, np.arange(512, dtype=np.float32) + delta)


@pytest.mark.parametrize(("dtype", "extent"), [(jnp.int8, 128), (jnp.uint8, 256)])
def test_nested_integer_iota_uses_its_window_extent(dtype, extent):
    def build(source, *, interpret=False):
        def kernel(x, y):
            y[:] = x[:]

            def body():
                y[:extent] = jnp.arange(extent, dtype=source).astype(jnp.float32)

            jax.lax.cond(pl.program_id(0) == 0, body, body)

        spec = pl.BlockSpec((512,), lambda i: (i,))
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((512,), jnp.float32),
            grid=(1,),
            in_specs=(spec,),
            out_specs=spec,
            interpret=interpret,
        )

    functions = (build(dtype), build(jnp.int32))
    assert (
        compare_functions(*functions, jax.ShapeDtypeStruct((512,), jnp.float32)).status
        is Status.EQUIVALENT
    )
    arguments = np.full(512, -1, np.float32)
    expected = arguments.copy()
    expected[:extent] = np.arange(extent, dtype=np.float32)
    for source in (dtype, jnp.int32):
        np.testing.assert_array_equal(build(source, interpret=True)(arguments), expected)


@pytest.mark.parametrize("entry", ["algebra", "functions"])
@pytest.mark.parametrize(
    ("number", "source_dtype", "target_dtype", "expected", "wrong"),
    [
        (2**54 + 2**30 + 1, jnp.int64, jnp.float32, 2**54 + 2**31, 2**54),
        (-(2**54 + 2**30 + 1), jnp.int64, jnp.float32, -(2**54 + 2**31), -(2**54)),
        (2**63 + 2**39 + 1, jnp.uint64, jnp.float32, 2**63 + 2**40, 2**63),
        (2**54 + 2**30, jnp.int64, jnp.float32, 2**54, 2**54 + 2**31),
        (2**54 + 3 * 2**30, jnp.int64, jnp.float32, 2**54 + 2**32, 2**54 + 2**31),
        (2**24 + 1, jnp.int32, jnp.float32, 2**24, 2**24 + 2),
        (2**54 + 2**46 + 1, jnp.int64, jnp.bfloat16, 2**54, 2**54 + 2**47),
        (2049, jnp.int64, jnp.float16, 2048, 2050),
    ],
    ids=["positive", "negative", "unsigned", "tie-even", "tie-odd", "int32", "bf16", "f16"],
)
def test_integer_constant_casts_preserve_source_precision(
    entry, number, source_dtype, target_dtype, expected, wrong
):
    # Keep x64 local to the trace, comparison and CPU execution.
    with jax.enable_x64():
        reference = np.asarray(number, dtype=source_dtype).astype(target_dtype).astype(np.float32)
        assert int(reference) == expected
        functions = tuple(
            integer_constant_cast(number, source_dtype, target_dtype, literal=literal)
            for literal in (None, expected, wrong)
        )
        for right, status in zip(
            functions[1:], (Status.EQUIVALENT, Status.INEQUIVALENT), strict=True
        ):
            result = (
                compare_algebraic_jaxprs(*(traced(fn, size=128) for fn in (functions[0], right)))
                if entry == "algebra"
                else compare_functions(
                    functions[0], right, jax.ShapeDtypeStruct((128,), jnp.float32)
                )
            )
            assert result.status is status
        arguments = np.zeros(128, np.float32)
        for literal, scalar in ((None, reference), (expected, reference), (wrong, wrong)):
            actual = integer_constant_cast(
                number, source_dtype, target_dtype, literal=literal, interpret=True
            )(arguments)
            np.testing.assert_array_equal(actual, np.full(128, scalar, np.float32))


@pytest.mark.parametrize("entry", ["algebra", "functions"])
@pytest.mark.parametrize("constant", [False, True])
def test_closed_integer_sum_casts_preserve_exact_values(entry, constant):
    def build(literal=None, *, interpret=False):
        identity = pointwise(lambda x: x, size=128, interpret=interpret)

        def outer(x):
            values = jnp.arange(128, dtype=jnp.int64)
            if constant:
                values = values * 0 + 1
            bias = values.sum().astype(jnp.float32) if literal is None else jnp.float32(literal)
            return identity(x + bias)

        return outer

    with jax.enable_x64():
        values = np.ones(128, np.int64) if constant else np.arange(128, dtype=np.int64)
        expected = values.sum().astype(np.float32)
        for literal, status in ((expected, Status.EQUIVALENT), (expected + 1, Status.INEQUIVALENT)):
            functions = (build(), build(literal))
            result = (
                compare_algebraic_jaxprs(*(traced(fn, size=128) for fn in functions))
                if entry == "algebra"
                else compare_functions(*functions, jax.ShapeDtypeStruct((128,), jnp.float32))
            )
            assert result.status is status
        arguments = np.arange(128, dtype=np.float32) / 4
        for literal, scalar in (
            (None, expected),
            (expected, expected),
            (expected + 1, expected + 1),
        ):
            np.testing.assert_array_equal(
                build(literal, interpret=True)(arguments), arguments + scalar
            )


@pytest.mark.parametrize(
    ("placement", "entry"),
    [("direct", "algebra"), ("outer", "algebra"), ("cond", "algebra"), ("outer", "functions")],
)
@pytest.mark.parametrize("different", [False, True])
def test_bounded_coordinate_casts_keep_arithmetic_and_program_offsets(placement, entry, different):
    left = coordinate_cast(True, placement=placement)
    right = coordinate_cast(False, placement=placement, offset=2 if different else 1)
    result = (
        compare_algebraic_jaxprs(*(traced(fn, size=256) for fn in (left, right)))
        if entry == "algebra"
        else compare_functions(left, right, jax.ShapeDtypeStruct((256,), jnp.float32))
    )
    assert result.status is (Status.INEQUIVALENT if different else Status.EQUIVALENT)
    arguments = np.arange(256, dtype=np.float32) / 4
    for before, offset in ((True, 1), (False, 2 if different else 1)):
        actual = coordinate_cast(before, placement=placement, offset=offset, interpret=True)(
            arguments
        )
        expected = arguments + (np.arange(256, dtype=np.int32) + offset).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("dtype", "precision"), [(jnp.bfloat16, 8), (jnp.float16, 11), (jnp.float32, 24)]
)
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("outside", [False, True])
def test_coordinate_cast_precision_bounds_are_closed_and_conservative(
    dtype, precision, sign, outside
):
    limit, size = 2**precision, 256
    base = limit - (1 if outside else size) if sign > 0 else -limit + (2 - size if outside else 1)
    functions = tuple(
        coordinate_cast(before, dtype=dtype, base=base, offset=sign) for before in (True, False)
    )
    sources = tuple(traced(fn, size=size) for fn in functions)
    if outside:
        with pytest.raises(MathematicalUndecidedError):
            compare_algebraic_jaxprs(*sources)
    else:
        assert compare_algebraic_jaxprs(*sources).status is Status.EQUIVALENT
    coordinates = np.arange(size, dtype=np.int32) + base
    expected = (
        (coordinates + sign).astype(dtype).astype(np.float32),
        (coordinates.astype(dtype) + np.asarray(sign, dtype=dtype)).astype(np.float32),
    )
    for before, reference in zip((True, False), expected, strict=True):
        actual = coordinate_cast(before, dtype=dtype, base=base, offset=sign, interpret=True)(
            np.zeros(size, np.float32)
        )
        np.testing.assert_array_equal(actual, reference)
    assert np.array_equal(*expected) == (not outside)


def test_integer_input_values_do_not_inherit_coordinate_cast_bounds():
    signature = jax.ShapeDtypeStruct((128,), jnp.int32)
    spec = pl.BlockSpec((128,), lambda p: (p,))

    def build(before, *, interpret=False):
        def kernel(x, y):
            y[:] = (x[:] + 1).astype(jnp.float32) if before else x[:].astype(jnp.float32) + 1

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
            grid=(1,),
            in_specs=(spec,),
            out_specs=spec,
            interpret=interpret,
        )

    with pytest.raises(MathematicalUndecidedError):
        compare_algebraic_jaxprs(
            *(jax.make_jaxpr(build(before))(signature) for before in (True, False))
        )
    arguments = np.arange(128, dtype=np.int32) + 2**24
    expected = ((arguments + 1).astype(np.float32), arguments.astype(np.float32) + 1)
    assert not np.array_equal(*expected)
    for before, reference in zip((True, False), expected, strict=True):
        np.testing.assert_array_equal(build(before, interpret=True)(arguments), reference)


@pytest.mark.parametrize("kind", ["sum", "dot", "nested_sum"])
@pytest.mark.parametrize("extent", [256, 512])
def test_coordinate_casts_inside_contractions_use_the_input_domain(kind, extent):
    size = 128

    def build(before):
        identity = pointwise(lambda x: x, size=size)

        def outer(x):
            coordinates = jnp.arange(extent, dtype=jnp.int32) + 2**24 - 256
            if kind == "nested_sum":
                coordinates = jnp.broadcast_to(coordinates, (2, extent))
            values = (
                (coordinates + 1).astype(jnp.float32)
                if before
                else coordinates.astype(jnp.float32) + 1
            )
            scalar = (
                jnp.dot(values, jnp.ones_like(values))
                if kind == "dot"
                else jnp.sum(jnp.sum(values, axis=1), axis=0)
                if kind == "nested_sum"
                else jnp.sum(values)
            )
            return identity(x + scalar)

        return outer

    sources = tuple(traced(build(before), size=size) for before in (True, False))
    if extent == 256:
        assert compare_algebraic_jaxprs(*sources).status is Status.EQUIVALENT
    else:
        with pytest.raises(MathematicalUndecidedError):
            compare_algebraic_jaxprs(*sources)


@pytest.mark.parametrize("candidate", ["commuted", "after"])
def test_closed_contraction_with_opaque_cast_can_be_narrowed(candidate):
    def build(mode, *, interpret=False):
        def operation(x):
            coordinates = jnp.arange(128, dtype=jnp.int32)
            value = (
                (coordinates + 1).astype(jnp.bfloat16)
                if mode == "before"
                else (1 + coordinates).astype(jnp.bfloat16)
                if mode == "commuted"
                else coordinates.astype(jnp.bfloat16) + jnp.asarray(1, jnp.bfloat16)
            )
            return jnp.sum(value.astype(jnp.float32)).astype(jnp.bfloat16)

        return pointwise(operation, size=128, dtype=jnp.bfloat16, interpret=interpret)

    result = compare_algebraic_jaxprs(
        *(traced(build(mode), size=128) for mode in ("before", candidate))
    )
    assert result.status is Status.EQUIVALENT
    expected = np.full(128, np.sum(np.arange(1, 129, dtype=np.float32)), dtype=jnp.bfloat16)
    for mode in ("before", candidate):
        np.testing.assert_array_equal(
            build(mode, interpret=True)(np.zeros(128, np.float32)), expected
        )


@pytest.mark.parametrize("entry", ["algebra", "functions"])
@pytest.mark.parametrize("size", [128, 256])
def test_outer_vector_coordinate_arithmetic_uses_its_shape_bounds(entry, size):
    def build(split, *, interpret=False):
        identity = pointwise(lambda x: x, size=size, interpret=interpret)

        def outer(x):
            offsets = jnp.arange(size, dtype=jnp.int32)
            offsets = (offsets + 1) + 2 if split else offsets + 3
            return identity(x + offsets.astype(jnp.float32))

        return outer

    functions = tuple(build(split) for split in (False, True))
    result = (
        compare_algebraic_jaxprs(*(traced(fn, size=size) for fn in functions))
        if entry == "algebra"
        else compare_functions(*functions, jax.ShapeDtypeStruct((size,), jnp.float32))
    )
    assert result.status is Status.EQUIVALENT
    arguments = np.arange(size, dtype=np.float32) / 4
    expected = arguments + (np.arange(size, dtype=np.int32) + 3).astype(np.float32)
    for split in (False, True):
        np.testing.assert_array_equal(build(split, interpret=True)(arguments), expected)


@pytest.mark.parametrize(
    ("condition", "expected_status"),
    [
        ("extent", Status.EQUIVALENT),
        ("singleton", Status.EQUIVALENT),
        ("block", Status.INEQUIVALENT),
        ("data", Status.INEQUIVALENT),
    ],
)
def test_outer_coordinate_bounds_do_not_become_block_or_data_bounds(condition, expected_status):
    size, block = 256, 128
    limit = {"extent": size, "singleton": 1, "block": block, "data": size}[condition]

    def build(*, interpret=False):
        identity = pointwise(lambda x: x, block, size=size, interpret=interpret)

        def outer(x):
            coordinate = jnp.arange(1 if condition == "singleton" else size, dtype=jnp.int32)
            predicate = (x if condition == "data" else coordinate) < limit
            return identity(jnp.where(predicate, x + 1, x + 2))

        return outer

    reference = pointwise(lambda x: x + 1, block, size=size)
    result = compare_algebraic_jaxprs(traced(build(), size=size), traced(reference, size=size))
    assert result.status is expected_status
    arguments = np.arange(size, dtype=np.float32) + block
    predicate = (
        arguments < size
        if condition == "data"
        else np.arange(size) < block
        if condition == "block"
        else True
    )
    expected = np.where(predicate, arguments + 1, arguments + 2)
    np.testing.assert_array_equal(build(interpret=True)(arguments), expected)
    assert np.array_equal(expected, arguments + 1) == (expected_status is Status.EQUIVALENT)


@pytest.mark.parametrize("placement", ["outer", "direct", "cond"])
def test_rank_one_broadcast_cannot_alias_a_full_coordinate_vector(placement):
    arguments = np.arange(256, dtype=np.float32)
    extent = 256 if placement == "outer" else 128
    left, right = (rank_one_broadcast(width, placement=placement) for width in (1, extent))
    sources = tuple(jax.make_jaxpr(fn)(arguments) for fn in (left, right))
    actual = [
        np.asarray(rank_one_broadcast(width, placement=placement, interpret=True)(arguments))
        for width in (1, extent)
    ]
    np.testing.assert_array_equal(actual[0], arguments + 3)
    np.testing.assert_array_equal(actual[1], arguments + np.arange(256) % extent + 3)
    assert not np.array_equal(*actual)
    assert compare_algebraic_jaxprs(*sources).status is Status.INEQUIVALENT


@pytest.mark.parametrize("source", ["iota", "input", "constant"])
@pytest.mark.parametrize("placement", ["outer", "direct", "cond"])
def test_rank_one_broadcast_matches_its_explicit_coordinate_map(source, placement):
    inputs = (jax.ShapeDtypeStruct((256,), jnp.float32),)
    if source == "input":
        inputs += (jax.ShapeDtypeStruct((1,), jnp.float32),)
    sources = tuple(
        jax.make_jaxpr(rank_one_broadcast(placement=placement, source=source, explicit=explicit))(
            *inputs
        )
        for explicit in (False, True)
    )
    assert compare_algebraic_jaxprs(*sources).status is Status.EQUIVALENT


@pytest.mark.parametrize("kind", ["sum", "max", "dot"])
@pytest.mark.parametrize("guarded", [False, True])
def test_rank_one_contractions_use_the_complete_domain_in_public_algebra(kind, guarded):
    left = traced(rank_one_reduction(kind, guarded=guarded), size=256)
    reordered = traced(rank_one_reduction(kind, guarded=guarded, reverse=True), size=256)
    different = traced(rank_one_reduction(kind, 256, guarded=guarded), size=256)
    assert compare_algebraic_jaxprs(left, reordered).status is Status.EQUIVALENT
    assert compare_algebraic_jaxprs(left, different).status is Status.INEQUIVALENT


@pytest.mark.parametrize("kind", ["sum", "max", "dot"])
def test_rank_one_outer_contractions_keep_their_bound_input_coordinates(kind):
    identity = pointwise(lambda x: x, size=256)

    def outer(x):
        scalar = {"sum": jnp.sum, "max": jnp.max, "dot": lambda v: jnp.dot(v, v)}[kind](x)
        return identity(jnp.broadcast_to(scalar, x.shape))

    same = traced(rank_one_reduction(kind, 256), size=256)
    different = traced(rank_one_reduction(kind, 128), size=256)
    source = traced(outer, size=256)
    assert compare_algebraic_jaxprs(source, same).status is Status.EQUIVALENT
    assert compare_algebraic_jaxprs(source, different).status is Status.INEQUIVALENT


def mathematical(a, b, **kwargs):
    assert compare_jaxprs(a, b).status is Status.DIFFERENT
    result = compare_algebraic_jaxprs(a, b, **kwargs)
    assert result.correspondence is None
    assert result.backend_result.backend == "sympy"
    assert result.backend_result.queries
    assert result.warnings and result.normalizations
    assert "blocked-memory-to-global-values:v2" in result.rules
    return result


@pytest.mark.parametrize("mode", ["add", "nonlinear"])
@pytest.mark.parametrize("reverse", [False, True])
def test_retiling_uses_complete_projection_and_trusted_cas(mode, reverse):
    result = mathematical(trace(128, mode=mode), trace(256, reverse, mode=mode))
    assert result.status is Status.EQUIVALENT
    assert all(p.projection.conditions for p in result.normalizations)


def test_tail_domain_is_symbolic_even_for_large_arrays():
    size = 2**24 + 13
    result = mathematical(
        trace(128, size=size, tail=True), trace(256, True, size=size, tail=True), algebra_budget=512
    )
    assert result.status is Status.EQUIVALENT
    assert result.work <= 512


@pytest.mark.parametrize("candidate", [lambda x: 2 * x + 2, lambda x: (x + 1) * 2])
def test_different_arithmetic_graphs_are_compared_after_memory_lowering(candidate):
    result = mathematical(
        traced(pointwise(lambda x: (x + 1) + (x + 1))), traced(pointwise(candidate, 256))
    )
    assert result.status is Status.EQUIVALENT


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_pallas_and_jax_share_ordered_external_inputs(swap, wrapped):
    pallas = pointwise(lambda x, y: (x - y) + (x - y), inputs=2)

    def ordinary(x, y):
        return 2 * (x - y)

    a, b = (jax.jit(pallas), jax.jit(ordinary)) if wrapped else (pallas, ordinary)
    if swap:
        a, b = b, a
    result = mathematical(traced(a, inputs=2), traced(b, inputs=2))
    assert result.status is Status.EQUIVALENT
    assert sum(p is not None for p in result.normalizations) == 1


def test_complete_output_difference_has_a_concrete_coordinate_query():
    result = mathematical(trace(), trace(mode="add_two"))
    assert result.status is Status.INEQUIVALENT and result.output == 0
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_multiple_outputs_cannot_be_swapped_or_partially_matched():
    p = pointwise(lambda x: (x + 1, x * 2), outputs=2)
    a = traced(p)
    assert mathematical(a, traced(lambda x: (1 + x, 2 * x))).status is Status.EQUIVALENT
    result = mathematical(a, traced(lambda x: (1 + x, 3 * x)))
    assert result.status is Status.INEQUIVALENT and result.output == 1
    assert mathematical(a, traced(lambda x: (2 * x, 1 + x))).status is Status.INEQUIVALENT


def test_cast_and_nonfinite_values_do_not_acquire_real_algebra_identities():
    def fn(x):
        return x.astype(jnp.bfloat16).astype(jnp.float32)

    assert mathematical(traced(pointwise(fn)), traced(fn)).status is Status.EQUIVALENT
    with pytest.raises(MathematicalUndecidedError):
        mathematical(traced(pointwise(fn)), traced(lambda x: x))
    with pytest.raises(MathematicalUndecidedError):
        mathematical(
            traced(pointwise(lambda x: jnp.log(x) - jnp.log(x))),
            traced(lambda x: jnp.zeros_like(x)),
        )


def test_sequential_duplicate_store_retains_the_latest_output_value():
    assert compare_algebraic_jaxprs(trace(mode="duplicate"), trace()).status is Status.EQUIVALENT
    assert (
        compare_algebraic_jaxprs(trace(mode="duplicate"), trace(mode="add_two")).status
        is Status.INEQUIVALENT
    )


@pytest.mark.parametrize("mode", ["uninitialized", "input_write", "partial"])
def test_effects_and_unsafe_memory_cannot_be_erased_by_algebra(mode):
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_algebraic_jaxprs(trace(mode=mode), trace())
    assert error.value.stage == "algebra"
    assert error.value.failure_kind == "unsupported_conversion"


def test_retained_unused_effect_is_checked_even_if_frontend_normally_eliminates_it():
    graph = trace()
    outer = graph.eqns[0]
    kernel = outer.params["jaxpr"]
    unused = kernel.eqns[0].replace(
        invars=[kernel.invars[1]],
        outvars=[core.DropVar(kernel.eqns[0].outvars[0].aval)],
        effects={st.ReadEffect(kernel.invars[1])},
    )
    kernel = kernel.replace(eqns=[unused, *kernel.eqns], effects=kernel.effects | unused.effects)
    changed = graph.replace(eqns=[outer.replace(params={**outer.params, "jaxpr": kernel})])
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_algebraic_jaxprs(changed, graph)


def test_program_id_used_as_data_is_not_erased_by_retiling():
    result = mathematical(trace(128, mode="program_id"), trace(256, mode="program_id"))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_equal_read_sets_do_not_erase_the_coordinate_to_value_mapping():
    result = mathematical(trace(256), trace(256, in_mapping=lambda p: (3 - p,)))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_regular_semaphore_is_quiescent_between_independent_programs():
    def kernel(x, y, sem):
        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1)
        y[:] = x[:] + 1

    sig = jax.ShapeDtypeStruct((1024,), jnp.float32)
    spec = pl.BlockSpec((128,), lambda i: (i,))
    fn = pl.pallas_call(
        kernel,
        out_shape=sig,
        grid=(8,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.REGULAR,),
    )
    a = traced(fn)
    assert compare_jaxprs(a, a).status is Status.EQUIVALENT
    result = compare_algebraic_jaxprs(a, trace())
    assert result.status is Status.EQUIVALENT
    assert "local-control-memory-to-values:v1" in result.rules


def test_local_dma_projection_retains_progress_assumption_and_wait_evidence():
    from tests.test_pallas_dma import copy, function, relayed

    for fn in (function(copy), function(relayed, arrays=1)):
        result = mathematical(traced(fn, size=128), traced(lambda x: x, size=128))
        assert result.status is Status.EQUIVALENT
        assert any("eventually completes" in text for text in result.assumptions)
        assert result.normalizations[0].projection.communication
        assert result.normalizations[0].projection.waits


def test_split_dma_preserves_piecewise_output_coordinates():
    from tests.test_pallas_dma import function

    def kernel(x, y, semaphore):
        tpu.async_copy(x.at[128:], y.at[:128], semaphore).wait()
        tpu.async_copy(x.at[:128], y.at[128:], semaphore).wait()

    a = traced(function(kernel, size=256, block=256), size=256)
    b = traced(lambda x: jnp.concatenate((x[128:], x[:128])), size=256)
    result = mathematical(a, b)
    assert result.status is Status.EQUIVALENT
    assert result.normalizations[0].projection.outputs[0].operation == "select_n"
    assert mathematical(a, traced(lambda x: x, size=256)).status is Status.INEQUIVALENT


def test_public_auto_mode_and_cpu_oracle():
    a, b = pointwise(lambda x: (x + 1) * 2), pointwise(lambda x: 2 * x + 2, 256)
    result = compare_functions(a, b, jax.ShapeDtypeStruct((1024,), jnp.float32))
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == "algebraic"
    x = jnp.linspace(-1, 1, 1024)
    for block, formula in ((128, lambda x: (x + 1) * 2), (256, lambda x: 2 * x + 2)):
        actual = pointwise(formula, block, interpret=True)(x)
        np.testing.assert_allclose(actual, 2 * np.asarray(x) + 2, rtol=2e-6, atol=2e-6)


def test_analysis_and_cas_share_the_public_algebra_budget():
    with pytest.raises(ComparisonBudgetError) as error:
        compare_algebraic_jaxprs(trace(128), trace(256), algebra_budget=2)
    assert error.value.stage == "algebra"


def test_nondefault_compiler_options_are_not_dropped():
    with pytest.raises(UnsupportedJaxprError):
        compare_algebraic_jaxprs(
            traced(build(128, compiler_params=tpu.CompilerParams(flags={"test_flag": True}))),
            trace(256),
        )


def test_shard_local_projection_cannot_be_used_as_global_outputs():
    from tests.test_pallas_sharding import sharded

    with pytest.raises(UnsupportedJaxprError, match="global shard assembly") as error:
        compare_algebraic_jaxprs(traced(sharded(128), size=2048), traced(sharded(256), size=2048))
    assert error.value.stage == "algebra"
    assert error.value.failure_kind == "unsupported_conversion"
