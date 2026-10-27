"""Flat scalar-prefetch None positions remain source facts after flattening."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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
    NormalizedEvidence,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.view_graph import match_views


def prefetch_function(*, optional=True, before=False, offset=1, interpret=False):
    def kernel(*args):
        value, output = args[-2:]
        output[:] = value[:] + offset

    spec = pl.BlockSpec((128,), lambda program, *prefetched: (program,))
    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2 if optional else 1,
            grid=(1,),
            in_specs=(spec,),
            out_specs=spec,
        ),
        interpret=interpret,
    )

    def call(value):
        prefetched = (jnp.asarray([0], dtype=jnp.int32),)
        if optional:
            prefetched = (None, *prefetched) if before else (*prefetched, None)
        return function(*prefetched, value)

    return call


def trace(**options):
    return jax.make_jaxpr(prefetch_function(**options))(jax.ShapeDtypeStruct((128,), jnp.float32))


def call_view(graph):
    return extract_call(adapt(graph).view, budget=AnalysisBudget()).operation.call


@pytest.mark.parametrize(
    "optional,before,bindings",
    [
        (False, False, (True, True)),
        (True, False, (True, True, False)),
        (True, True, (True, False, True)),
    ],
)
def test_none_positions_are_retained_and_interpreter_values_agree(optional, before, bindings):
    graph = trace(optional=optional, before=before)
    assert call_view(graph).index_map_bindings == bindings
    value = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(
        np.asarray(prefetch_function(optional=optional, before=before, interpret=True)(value)),
        value + 1,
    )


@pytest.mark.parametrize("scope", ["function", "device_local"])
@pytest.mark.parametrize("before", [False, True])
def test_flattened_values_can_match_without_losing_distinct_tree_layouts(scope, before):
    graphs = trace(optional=False), trace(before=before)
    assert compare_jaxprs(*graphs, include_derivative_rules=False).status is Status.DIFFERENT
    assert compare_pallas_kernels(*graphs).status is Status.DIFFERENT
    assert match_views(*(adapt(graph).view for graph in graphs)) is None
    result = compare_algebraic_jaxprs(*graphs, scope=scope)
    assert result.status is Status.EQUIVALENT
    assert tuple(proof.index_map_bindings for proof in result.normalizations) == (
        (True, True),
        (True, False, True) if before else (True, True, False),
    )


@pytest.mark.parametrize("scope", ["function", "device_local"])
def test_none_positions_do_not_hide_changed_kernel_values(scope):
    graphs = trace(), trace(offset=2)
    assert compare_algebraic_jaxprs(*graphs, scope=scope).status is Status.INEQUIVALENT
    value = np.arange(128, dtype=np.float32)
    assert not np.array_equal(
        prefetch_function(interpret=True)(value),
        prefetch_function(offset=2, interpret=True)(value),
    )


@pytest.mark.parametrize("forged_bindings", [(True, False, True), (1, 1, 0)])
def test_normalization_replays_the_original_none_positions(forged_bindings):
    graphs = trace(optional=False), trace()
    views = tuple(adapt(graph).view for graph in graphs)
    proofs = tuple(normalize_view(view, budget=AnalysisBudget()) for view in views)
    evidence = NormalizedEvidence(*proofs, ((0, 0),))
    verify_normalized_evidence(*views, evidence, budget=AnalysisBudget())
    forged = replace(
        evidence,
        right=replace(evidence.right, index_map_bindings=forged_bindings),
    )
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(*views, forged, budget=AnalysisBudget())
    local = compare_pallas_kernels(*graphs, equivalence="semantic")
    verify_pallas_kernel_comparison(*graphs, local)
    forged_local = replace(
        local, right_protocol=replace(local.right_protocol, index_map_bindings=forged_bindings)
    )
    with pytest.raises(CorrespondenceError, match="protocol evidence"):
        verify_pallas_kernel_comparison(*graphs, forged_local)


@pytest.mark.parametrize(
    "layout",
    [
        ((None, 0, 0), {}),
        ((0, (0,)), {}),
        ((0, [0]), {}),
        ((0, {"value": 0}), {}),
        ((0, None), {}),
        ((0, 0, 0), {}),
        ((0, 0), {"keyword": None}),
        ([0, 0], {}),
        ((0, 0),),
    ],
)
def test_index_map_bindings_reject_nonflat_or_inconsistent_source_layout(layout):
    graph = trace()
    (operation,) = graph.eqns
    mapping = replace(operation.params["grid_mapping"], index_map_tree=jax.tree.structure(layout))
    forged = graph.replace(
        eqns=[operation.replace(params={**operation.params, "grid_mapping": mapping})]
    )
    with pytest.raises(UnsupportedJaxprError, match="index map bindings"):
        adapt(forged)
