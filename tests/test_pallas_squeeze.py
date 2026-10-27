"""Squeeze retains tensor coordinates, input provenance, and definedness."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl

from jax_reorder import (
    InvalidJaxprError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.memory import ValueExpr, _substitute, scalar_type
from jax_reorder.core.model import ParameterView
from jax_reorder.core.tensor_domain import coordinate, requires_tensor_coordinates
from jax_reorder.core.tensor_values import tensor_value
from tests.pallas_projection_oracle import at_point


def squeeze_call(shape, dimensions, *, placement="kernel", mode="squeeze", interpret=False):
    target = tuple(n for i, n in enumerate(shape) if i not in dimensions)

    def view(value):
        if mode == "reshape":
            return lax.reshape(value, target)
        if mode == "transpose":
            kept = tuple(i for i in range(len(shape)) if i not in dimensions)
            transposed = lax.transpose(value, kept + dimensions)
            return lax.squeeze(transposed, tuple(range(len(kept), len(shape))))
        return lax.squeeze(value, dimensions)

    def kernel(x, other, out):
        value = other[...] if mode == "wrong_source" else x[...]
        if placement == "kernel":
            value = view(value)
        out[...] = value + 1 if mode == "wrong_value" else value

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(target, jnp.float32),
        interpret=interpret,
    )
    if placement == "outer":
        return lambda x, other: call(view(x), view(other))
    return call


@pytest.mark.parametrize("placement", ["kernel", "outer"])
@pytest.mark.parametrize("scope", ["function", "device_local"])
@pytest.mark.parametrize(
    "shape,dimensions", [((1, 8, 128), (0,)), ((8, 1, 128), (1,)), ((1, 8, 1, 128), (2, 0))]
)
def test_squeeze_commutes_with_unit_axis_permutation(shape, dimensions, placement, scope):
    signature = jax.ShapeDtypeStruct(shape, jnp.float32)
    graphs = tuple(
        jax.make_jaxpr(squeeze_call(shape, dimensions, placement=placement, mode=mode))(
            signature, signature
        )
        for mode in ("squeeze", "transpose")
    )
    result = compare_algebraic_jaxprs(*graphs, scope=scope)
    assert result.status is Status.EQUIVALENT
    if scope == "device_local":
        assert result.scope == "device_local"
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = squeeze_call(shape, dimensions, placement=placement, interpret=True)(x, x + 5)
    np.testing.assert_array_equal(np.asarray(actual), np.squeeze(x, axis=dimensions))


def test_multiaxis_squeeze_and_general_reshape_coordinate_identity_remains_undecided():
    # Existing reshape normalization emits div/mod of a flattened coordinate;
    # this boundary is separate from the new direct squeeze coordinate map.
    shape, dimensions = (1, 8, 128), (0,)
    signature = jax.ShapeDtypeStruct(shape, jnp.float32)
    graphs = tuple(
        jax.make_jaxpr(squeeze_call(shape, dimensions, mode=mode))(signature, signature)
        for mode in ("squeeze", "reshape")
    )
    with pytest.raises(MathematicalUndecidedError, match="SymPy did not establish"):
        compare_algebraic_jaxprs(*graphs, algebra_timeout=10)


@pytest.mark.parametrize("mode", ["wrong_source", "wrong_value"])
@pytest.mark.parametrize("placement", ["kernel", "outer"])
@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_squeeze_does_not_erase_input_identity_or_changed_values(mode, placement, scope):
    shape, dimensions = (1, 8, 128), (0,)
    signature = jax.ShapeDtypeStruct(shape, jnp.float32)
    graphs = tuple(
        jax.make_jaxpr(squeeze_call(shape, dimensions, placement=placement, mode=variant))(
            signature, signature
        )
        for variant in ("squeeze", mode)
    )
    assert compare_algebraic_jaxprs(*graphs, scope=scope).status is Status.INEQUIVALENT
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = squeeze_call(shape, dimensions, placement=placement, mode=mode, interpret=True)(
        x, x + 5
    )
    delta = 5 if mode == "wrong_source" else 1
    np.testing.assert_array_equal(np.asarray(actual), np.squeeze(x, axis=dimensions) + delta)


def squeeze_view(shape=(1, 2, 1, 3), dimensions=(0, 2)):
    graph = jax.make_jaxpr(lambda x: lax.squeeze(x, dimensions))(
        jax.ShapeDtypeStruct(shape, jnp.float32)
    )
    return graph, adapt(graph).view


@pytest.mark.parametrize(
    "shape,dimensions", [((1,), (0,)), ((1, 2, 1, 3), (2, 0)), ((1, 0, 3), (0,))]
)
def test_squeeze_maps_removed_coordinates_to_zero_without_flattening(shape, dimensions):
    _, view = squeeze_view(shape, dimensions)
    (op,) = view.root.operations
    typ = view.root.inputs[0].type
    assert requires_tensor_coordinates("squeeze", (typ,), (op.results[0].type,))
    source = ValueExpr(
        "input", scalar_type(typ), tuple(coordinate("local", i) for i in range(len(shape))), input=0
    )
    projected = tensor_value(op, (source,), (typ,), {}, {}, (), AnalysisBudget())
    projected = _substitute(
        projected,
        {f"local:{i}": coordinate("output", i) for i in range(len(op.results[0].type.shape))},
    )
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = np.squeeze(x, axis=dimensions)
    actual = np.array([at_point(projected, (x,), p) for p in np.ndindex(expected.shape)]).reshape(
        expected.shape
    )
    np.testing.assert_array_equal(actual, expected)
    for dimension in dimensions:
        assert projected.operands[dimension].kind == "constant"
        assert projected.operands[dimension].value == 0


@pytest.mark.parametrize("dimensions", [(0, 0), (-1,), (4,), (1,), (True,), (0.0,), None, [0, 2]])
def test_squeeze_rejects_invalid_dimensions_in_typed_view(dimensions):
    _, view = squeeze_view()
    (op,) = view.root.operations
    op = replace(op, parameters=(ParameterView("dimensions", dimensions),))
    with pytest.raises(UnsupportedJaxprError, match="distinct unit axes"):
        tensor_value(
            op, (coordinate("local", 1),), (view.root.inputs[0].type,), {}, {}, (), AnalysisBudget()
        )


@pytest.mark.parametrize("invalid", ["shape", "dtype", "weak_type", "missing", "extra", "arity"])
def test_squeeze_checks_complete_typed_contract(invalid):
    _, view = squeeze_view()
    (op,) = view.root.operations
    types, args = (view.root.inputs[0].type,), (coordinate("local", 1),)
    if invalid in {"shape", "dtype", "weak_type"}:
        change = {"shape": (3, 2), "dtype": "int32", "weak_type": True}
        typ = replace(op.results[0].type, **{invalid: change[invalid]})
        op = replace(op, results=(replace(op.results[0], type=typ),))
    elif invalid == "missing":
        op = replace(op, parameters=())
    elif invalid == "extra":
        op = replace(op, parameters=op.parameters + (ParameterView("ignored", 1),))
    else:
        types, args = types * 2, args * 2
    with pytest.raises(UnsupportedJaxprError, match="squeeze"):
        tensor_value(op, args, types, {}, {}, (), AnalysisBudget())


def test_squeeze_adapter_rejects_malformed_source_axes():
    graph, _ = squeeze_view()
    op = graph.eqns[0].replace(params={"dimensions": (1,)})
    forged = graph.replace(jaxpr=graph.jaxpr.replace(eqns=[op]))
    with pytest.raises(InvalidJaxprError, match="size not equal to one"):
        adapt(forged)


@pytest.mark.parametrize("placement", ["kernel", "outer"])
@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_squeeze_cannot_cancel_partial_function_conditions(placement, scope):
    def function(drop):
        def kernel(x, out):
            value = x[...] if placement == "outer" else lax.squeeze(jnp.log(x[...]), (0,))
            out[...] = jnp.zeros((8, 128), dtype=jnp.float32) if drop else value * 0

        call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32))
        if placement == "outer":
            return lambda x: call(lax.squeeze(jnp.log(x), (0,)))
        return call

    signature = jax.ShapeDtypeStruct((1, 8, 128), jnp.float32)
    graphs = tuple(jax.make_jaxpr(function(drop))(signature) for drop in (False, True))
    with pytest.raises(MathematicalUndecidedError, match="conditions"):
        compare_algebraic_jaxprs(*graphs, scope=scope, algebra_timeout=10)
