"""KDA frontend coverage, with independent lane, value and resource checks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonContractError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt, parameter
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import ParameterView
from jax_reorder.core.normalization import normalize_view
from jax_reorder.core.tensor_values import tensor_value
from tests.pallas_projection_oracle import at_point
from tests.test_pallas_algebra import pointwise, traced


def identity(shape, *, memory_space=None, compiler=None, interpret=False):
    def kernel(x, y):
        y[...] = x[...]

    spec = pl.BlockSpec(memory_space=memory_space)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=compiler,
        interpret=interpret,
    )


PAD_CASES = (
    ((3,), ((2, 1, 0),)),
    ((4,), ((-1, 2, 1),)),
    ((3, 4), ((1, 0, 1), (-1, 2, 0))),
    ((2, 0), ((0, 0, 0), (1, 2, 3))),
)


@pytest.mark.parametrize("shape,padding", PAD_CASES)
@pytest.mark.parametrize("fill", [0.0, 7.0])
def test_static_pad_projects_edges_crops_and_interior_to_original_lanes(shape, padding, fill):
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 3
    expected = np.asarray(lax.pad(x, jnp.float32(fill), padding))
    call = identity(expected.shape)
    graph = jax.make_jaxpr(lambda x: call(lax.pad(x, jnp.float32(fill), padding)))(x)
    proof = normalize_view(adapt(graph).view, budget=AnalysisBudget())
    for point in np.ndindex(expected.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]
    # The independent native CPU interpreter must agree with the coordinate map.
    np.testing.assert_array_equal(identity(expected.shape, interpret=True)(expected), expected)


def test_padding_fill_is_preserved_in_mathematical_comparison():
    signature = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    call = identity((4, 5))

    def pad(x):
        return call(jnp.pad(x, ((1, 1), (1, 1)), constant_values=7))

    def concatenate(x, fill):
        row = jnp.full((1, 5), fill, jnp.float32)
        column = jnp.full((2, 1), fill, jnp.float32)
        return call(jnp.concatenate((row, jnp.concatenate((column, x, column), axis=1), row)))

    for fill, status in ((7, Status.EQUIVALENT), (8, Status.INEQUIVALENT)):
        graphs = (
            jax.make_jaxpr(pad)(signature),
            jax.make_jaxpr(lambda x, fill=fill: concatenate(x, fill))(signature),
        )
        assert compare_algebraic_jaxprs(*graphs).status is status


@pytest.mark.parametrize("invalid", [((0, 0, -1),), ((True, 0, 0),), ((0.0, 0, 0),)])
def test_pad_rejects_noninteger_or_negative_interior_schema(invalid):
    with pytest.raises(UnsupportedJaxprError, match="padding configuration"):
        parameter("padding_config", invalid, "pad")


def test_pad_projection_rechecks_the_output_shape():
    view = adapt(
        jax.make_jaxpr(lambda x: lax.pad(x, jnp.float32(0), ((1, 1, 0),)))(
            jax.ShapeDtypeStruct((3,), jnp.float32)
        )
    ).view
    (op,) = view.root.operations
    op = replace(op, parameters=(ParameterView("padding_config", ((2, 1, 0),)),))
    with pytest.raises(UnsupportedJaxprError, match="output shape"):
        tensor_value(
            op,
            (False, False),
            (view.root.inputs[0].type, op.operands[1].type),
            {},
            {},
            (),
            AnalysisBudget(),
        )


@pytest.mark.parametrize("scope", ["function", "device_local"])
@pytest.mark.parametrize("offset", [1.0, 2.0])
def test_exp2_real_identity_and_wrong_scale(scope, offset):
    left = pointwise(lambda x: lax.exp2(x + offset), size=128)
    right = pointwise(lambda x: 2 * lax.exp2(x), size=128)
    result = compare_algebraic_jaxprs(traced(left, size=128), traced(right, size=128), scope=scope)
    assert result.status is (Status.EQUIVALENT if offset == 1 else Status.INEQUIVALENT)
    assert "exp2" in result.rules


def test_matching_highest_precision_matmul_allows_real_scalar_relocation():
    signature = jax.ShapeDtypeStruct((4, 4), jnp.float32)

    def function(scale_inside):
        def kernel(a, b, y):
            lhs = 2 * a[...] if scale_inside else a[...]
            product = jnp.matmul(lhs, b[...], precision=lax.Precision.HIGHEST)
            y[...] = product if scale_inside else 2 * product

        return pl.pallas_call(kernel, out_shape=signature)

    graphs = [
        jax.make_jaxpr(function(scale_inside))(signature, signature)
        for scale_inside in (False, True)
    ]
    assert compare_pallas_kernels(*graphs, equivalence="semantic").status is Status.DIFFERENT
    result = compare_algebraic_jaxprs(*graphs, scope="device_local")
    assert result.status is Status.EQUIVALENT and result.warnings
    assert "pallas.paired-highest-precision:v1" in result.rules


@pytest.mark.parametrize("memory", [None, tpu.VMEM])
def test_explicit_vmem_blocks_use_the_buffered_value_model(memory):
    signature = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    source = jax.make_jaxpr(identity(signature.shape, memory_space=memory))(signature)
    reference = jax.make_jaxpr(identity(signature.shape))(signature)
    result = compare_pallas_kernels(source, reference, equivalence="semantic")
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, reference, result)


def test_vmem_admission_does_not_silently_accept_smem():
    source = jax.make_jaxpr(identity((8, 128), memory_space=tpu.SMEM))(
        jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
    with pytest.raises(UnsupportedJaxprError, match="Ref storage"):
        adapt(source)


def chained_index(mode, *, interpret=False):
    def kernel(x, y):
        def body(i, _):
            row = x.at[i]
            value = (
                row.at[pl.ds(1, 2), pl.ds(2, 4)][...]
                if mode == "chained"
                else x[i, pl.ds(1 if mode != "wrong" else 2, 2), pl.ds(2, 4)]
            )
            y[i] = value
            return None

        lax.fori_loop(0, 2, body, None)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 2, 4), jnp.float32),
        interpret=interpret,
    )


def test_scalar_then_slice_ref_chain_retains_axis_and_offset_binding():
    signature = jax.ShapeDtypeStruct((2, 4, 8), jnp.float32)
    source = jax.make_jaxpr(chained_index("chained"))(signature)
    for mode, expected in (("direct", Status.EQUIVALENT), ("wrong", Status.DIFFERENT)):
        result = compare_pallas_kernels(
            source, jax.make_jaxpr(chained_index(mode))(signature), equivalence="semantic"
        )
        assert result.status is expected
    x = np.arange(64, dtype=np.float32).reshape(signature.shape)
    np.testing.assert_array_equal(chained_index("chained", interpret=True)(x), x[:, 1:3, 2:6])


def dynamic_slice_chain(mode, *, interpret=False):
    def kernel(x, y):
        def body(i, _):
            if mode in ("chained", "outside"):
                # "outside" has a valid final row at i=0, but its intermediate
                # two-row view already leaves storage. Do not erase that bound.
                offset = 3 if mode == "outside" else 0
                selected = 0 if mode == "outside" else 1
                value = x.at[pl.ds(i + offset, 2), :].at[selected][...]
            else:
                value = x[i + (0 if mode == "wrong" else 1)]
            y[i] = value
            return None

        lax.fori_loop(0, 2, body, None)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 8), jnp.float32),
        compiler_params=tpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
    )


def test_dynamic_slice_then_scalar_chain_retains_intermediate_bounds():
    signature = jax.ShapeDtypeStruct((4, 8), jnp.float32)
    source = jax.make_jaxpr(dynamic_slice_chain("chained"))(signature)
    for mode, expected in (("direct", Status.EQUIVALENT), ("wrong", Status.DIFFERENT)):
        other = jax.make_jaxpr(dynamic_slice_chain(mode))(signature)
        assert compare_pallas_kernels(source, other, equivalence="semantic").status is expected
    outside = jax.make_jaxpr(dynamic_slice_chain("outside"))(signature)
    with pytest.raises(UnsupportedJaxprError, match="outside its intermediate view"):
        compare_pallas_kernels(outside, outside, equivalence="semantic")
    x = np.arange(32, dtype=np.float32).reshape(signature.shape)
    np.testing.assert_array_equal(dynamic_slice_chain("chained", interpret=True)(x), x[1:3])


@pytest.mark.parametrize("options", [{"disable_bounds_checks": True}, {"vmem_limit_bytes": 65536}])
@pytest.mark.parametrize("layout", [True, False])
def test_resource_compiler_hints_require_the_same_full_access_proof(options, layout):
    compiler = tpu.CompilerParams(**options, needs_layout_passes=layout)
    source = jax.make_jaxpr(identity((8, 128), compiler=compiler))(
        jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
    view = adapt(source).view
    assert view.root.operations[0].call.compiler.reviewed_resource_options
    result = compare_pallas_kernels(source, source, equivalence="semantic")
    assert result.status is Status.EQUIVALENT and result.protocol_checked
    verify_pallas_kernel_comparison(source, source, result)

    def invalid(x, y, scratch):
        y[...] = scratch[...]

    unsafe = pl.pallas_call(
        invalid,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        scratch_shapes=(tpu.VMEM((8, 128), jnp.float32),),
        compiler_params=compiler,
    )
    graph = jax.make_jaxpr(unsafe)(jax.ShapeDtypeStruct((8, 128), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_pallas_kernels(graph, graph, equivalence="semantic")

    default = jax.make_jaxpr(identity((8, 128)))(jax.ShapeDtypeStruct((8, 128), jnp.float32))
    # Capacity/check settings remain part of the public source boundary even
    # after both resource projections have succeeded.
    assert (
        compare_pallas_kernels(source, default, equivalence="semantic").status is Status.DIFFERENT
    )
    with pytest.raises(ComparisonContractError, match="local inputs"):
        compare_algebraic_jaxprs(source, default, scope="device_local")


@pytest.mark.parametrize("layout", [True, False])
def test_disabling_runtime_bounds_checks_still_rejects_out_of_bounds_access(layout):
    def kernel(x, y):
        y[...] = jnp.broadcast_to(x[8, 0], y.shape)

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        compiler_params=tpu.CompilerParams(disable_bounds_checks=True, needs_layout_passes=layout),
    )
    source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((8, 128), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="out of bounds|outside"):
        compare_pallas_kernels(source, source, equivalence="semantic")
