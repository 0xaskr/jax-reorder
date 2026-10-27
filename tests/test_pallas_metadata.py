"""Preserve diagnostic call metadata without making it an array-value contract."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.frozen_dict import FrozenDict
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    verify_normalized_evidence,
)
from jax_reorder.core.view_graph import match_views
from tests.pallas_remote_cases import exchange


def metadata_function(metadata, *, interpret=False):
    def kernel(x, y):
        y[...] = x[...] + 1

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        metadata=metadata,
        interpret=interpret,
    )


def trace(metadata):
    return jax.make_jaxpr(metadata_function(metadata))(jax.ShapeDtypeStruct((128,), jnp.float32))


def encoded(metadata):
    return None if metadata is None else tuple(sorted(metadata.items()))


@pytest.mark.parametrize(
    "metadata", [None, {}, {"xprof_metadata": '{ "block_q": 128, "note": "单位轴" }'}]
)
def test_metadata_adapts_without_parsing_or_dropping_strings(metadata):
    source = trace(metadata)
    view = adapt(source).view
    call = extract_call(view, budget=AnalysisBudget()).operation.call
    assert call.metadata == encoded(metadata)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    assert result.left_protocol.call_metadata == encoded(metadata)
    verify_pallas_kernel_comparison(source, source, result)
    x = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(metadata_function(metadata, interpret=True)(x)), x + 1)


@pytest.mark.parametrize(
    "left,right",
    [
        (None, {}),
        ({}, {"xprof_metadata": "first"}),
        ({"xprof_metadata": "first"}, {"xprof_metadata": "second"}),
    ],
)
@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_changed_metadata_is_structurally_different_but_keeps_the_same_values(left, right, scope):
    graphs = trace(left), trace(right)
    assert compare_jaxprs(*graphs, include_derivative_rules=False).status is Status.DIFFERENT
    assert compare_pallas_kernels(*graphs).status is Status.DIFFERENT
    views = tuple(adapt(graph).view for graph in graphs)
    assert match_views(*views) is None
    result = compare_algebraic_jaxprs(*graphs, scope=scope)
    assert result.status is Status.EQUIVALENT
    assert tuple(proof.call_metadata for proof in result.normalizations) == (
        encoded(left),
        encoded(right),
    )
    assert any(
        "metadata" in assumption and "diagnostic" in assumption for assumption in result.assumptions
    )


def test_metadata_dictionary_order_is_not_a_new_semantic_field():
    left = trace({"z": "last", "a": "first"})
    right = trace({"a": "first", "z": "last"})
    assert compare_jaxprs(left, right, include_derivative_rules=False).status is Status.EQUIVALENT
    assert compare_pallas_kernels(left, right).status is Status.EQUIVALENT


@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_computed_call_input_preserves_source_metadata(scope):
    metadata = {"xprof_metadata": "computed input"}

    def outer(description):
        def call(value):
            return metadata_function(description)(value * 2)

        return call

    graphs = tuple(
        jax.make_jaxpr(outer(description))(jax.ShapeDtypeStruct((128,), jnp.float32))
        for description in (None, metadata)
    )
    result = compare_algebraic_jaxprs(*graphs, scope=scope)
    assert result.status is Status.EQUIVALENT
    assert tuple(proof.call_metadata for proof in result.normalizations) == (
        None,
        encoded(metadata),
    )
    if scope == "device_local":
        local = compare_pallas_kernels(*graphs, equivalence="semantic")
        verify_pallas_kernel_comparison(*graphs, local)


def remote_metadata_graph(metadata):
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    spec = pl.BlockSpec((128,), lambda program: (program,))
    function = pl.pallas_call(
        exchange,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(pltpu.SemaphoreType.DMA,) * 2,
        metadata=metadata,
    )
    sharded = jax.shard_map(function, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)
    return jax.make_jaxpr(sharded)(jax.ShapeDtypeStruct((256,), jnp.float32))


@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_remote_normalization_preserves_metadata_and_replays_it(scope):
    graphs = remote_metadata_graph(None), remote_metadata_graph({"xprof_metadata": "remote"})
    result = compare_algebraic_jaxprs(*graphs, scope=scope)
    assert result.status is Status.EQUIVALENT
    assert tuple(proof.call_metadata for proof in result.normalizations) == (
        None,
        (("xprof_metadata", "remote"),),
    )
    if scope == "function":
        views = tuple(adapt(graph).view for graph in graphs)
        result = compare_program_views(*views)
        verify_normalized_evidence(*views, result.evidence, budget=AnalysisBudget())
        forged = replace(result.evidence, right=replace(result.evidence.right, call_metadata=None))
        with pytest.raises(CorrespondenceError, match="does not replay"):
            verify_normalized_evidence(*views, forged, budget=AnalysisBudget())
    else:
        local = compare_pallas_kernels(*graphs, equivalence="semantic")
        verify_pallas_kernel_comparison(*graphs, local)
        forged = replace(local, right_protocol=replace(local.right_protocol, call_metadata=None))
        with pytest.raises(CorrespondenceError, match="protocol evidence"):
            verify_pallas_kernel_comparison(*graphs, forged)


@pytest.mark.parametrize("side", ["left", "right"])
def test_normalized_certificate_replays_exact_source_metadata(side):
    views = tuple(adapt(trace(metadata)).view for metadata in (None, {"xprof_metadata": "value"}))
    result = compare_program_views(*views)
    assert result.status is Status.EQUIVALENT and result.evidence.kind == "normalized"
    verify_normalized_evidence(*views, result.evidence, budget=AnalysisBudget())
    proof = replace(getattr(result.evidence, side), call_metadata=(("forged", "value"),))
    forged = replace(result.evidence, **{side: proof})
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(*views, forged, budget=AnalysisBudget())


def test_local_certificate_replays_exact_source_metadata():
    left, right = trace(None), trace({"xprof_metadata": "value"})
    result = compare_pallas_kernels(left, right, equivalence="semantic")
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(left, right, result)
    forged = replace(
        result, left_protocol=replace(result.left_protocol, call_metadata=(("forged", "value"),))
    )
    with pytest.raises(CorrespondenceError, match="protocol evidence"):
        verify_pallas_kernel_comparison(left, right, forged)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"key": "value"},
        FrozenDict({"key": 3}),
        FrozenDict({3: "value"}),
        FrozenDict({"key": ("value",)}),
        FrozenDict({"key": object()}),
    ],
)
def test_metadata_requires_exact_locked_string_mapping(metadata):
    source = trace(None)
    (op,) = source.eqns
    forged = source.replace(eqns=[op.replace(params={**op.params, "metadata": metadata})])
    with pytest.raises(UnsupportedJaxprError, match="FrozenDict of string keys and values"):
        adapt(forged)


def test_reserved_mesh_axes_metadata_is_not_assumed_to_be_unconditionally_safe():
    with pytest.raises(UnsupportedJaxprError, match="reserved mesh_axes"):
        adapt(trace({"mesh_axes": '["data"]'}))
