"""Observable parent views retain full memory and communication obligations."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh, PartitionSpec

from jax_reorder import (
    ComparisonContractError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from tests.pallas_projection_oracle import at_point


def tail_graph(*, rows=16, wait=True, overwritten=False, consumed=False):
    def kernel(x, y, temporary, semaphore):
        temporary[:16, :] = x[...]
        if consumed:
            unused = temporary[...]
            temporary[...] = unused * 0
            temporary[:16, :] = x[...]
        descriptor = tpu.async_copy(temporary, y, semaphore)
        if overwritten:
            temporary[:16, :] = x[...]
        if wait:
            descriptor.wait()

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((32, 128), jnp.float32),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=(tpu.VMEM((32, 128), jnp.float32), tpu.SemaphoreType.DMA),
    )
    return jax.make_jaxpr(lambda x: function(x)[:rows].T)(
        jax.ShapeDtypeStruct((16, 128), jnp.float32)
    )


def test_parent_crop_proves_defined_lanes_and_keeps_the_full_dma():
    graph = tail_graph()
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_pallas_kernels(graph, graph, equivalence="semantic")
    result = compare_pallas_kernels(graph, graph, equivalence="semantic", output_boundary="parent")
    assert result.status is Status.EQUIVALENT and result.protocol_checked
    proof = result.left_protocol
    assert proof is not None and proof.output_boundary == "parent"
    assert len(proof.projection.communication) == len(proof.projection.waits) == 1
    assert proof.boundaries[0].outputs[0].shape == (128, 16)
    x = np.arange(16 * 128, dtype=np.float32).reshape(16, 128)
    for i, j in ((0, 0), (127, 15), (25, 7)):
        assert at_point(proof.projection.outputs[0], (x,), (i, j)) == x[j, i]
    verify_pallas_kernel_comparison(graph, graph, result)
    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(
            graph, graph, replace(result, left_protocol=replace(proof, output_bindings=()))
        )


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"rows": 17}, "uninitialized"),
        ({"wait": False}, "DMA|transfer"),
        ({"overwritten": True}, "DMA|flight|transfer"),
        ({"consumed": True}, "uninitialized"),
    ],
)
def test_parent_views_do_not_hide_observable_or_protocol_errors(options, reason):
    graph = tail_graph(**options)
    with pytest.raises(UnsupportedJaxprError, match=reason):
        compare_pallas_kernels(graph, graph, equivalence="semantic", output_boundary="parent")


def discarded_function(*, swap=False, arithmetic=False):
    signature = jax.ShapeDtypeStruct((4, 4), jnp.float32)

    def kernel(x, y, placeholder):
        y[...] = x[...]

    call = pl.pallas_call(kernel, out_shape=(signature, signature))

    def function(x):
        y, _ = call(x)
        if arithmetic:
            return y + 1
        return y.T if swap else y

    return function, signature


def test_discarded_outputs_require_explicit_parent_mode_and_replay():
    function, signature = discarded_function()
    graph = jax.make_jaxpr(function)(signature)
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_pallas_kernels(graph, graph, equivalence="semantic")
    for compare in (
        lambda: compare_pallas_kernels(graph, graph, output_boundary="parent"),
        lambda: compare_algebraic_jaxprs(
            graph, graph, scope="device_local", output_boundary="parent"
        ),
        lambda: compare_functions(
            function, function, signature, scope="device_local", output_boundary="parent"
        ),
    ):
        assert compare().status is Status.EQUIVALENT


def test_identical_kernel_does_not_bypass_different_parent_views():
    graphs = [
        jax.make_jaxpr(function)(signature)
        for function, signature in (discarded_function(), discarded_function(swap=True))
    ]
    assert compare_pallas_kernels(*graphs).status is Status.EQUIVALENT
    assert compare_pallas_kernels(*graphs, output_boundary="parent").status is Status.DIFFERENT
    result = compare_algebraic_jaxprs(*graphs, scope="device_local", output_boundary="parent")
    assert result.status is Status.INEQUIVALENT


def test_parent_arithmetic_requires_a_separate_reviewed_rule():
    function, signature = discarded_function(arithmetic=True)
    graph = jax.make_jaxpr(function)(signature)
    with pytest.raises(UnsupportedJaxprError, match="unreviewed parent"):
        compare_pallas_kernels(graph, graph, output_boundary="parent")
    with pytest.raises(ValueError, match="device_local"):
        compare_algebraic_jaxprs(graph, graph, output_boundary="parent")


def test_parent_rank_one_grid_views_preserve_ordered_concatenation_and_broadcast():
    signature = jax.ShapeDtypeStruct((256,), jnp.float32)
    spec = pl.BlockSpec((128,), lambda i: (i,))

    def kernel(x, y):
        y[...] = x[...]

    call = pl.pallas_call(kernel, out_shape=signature, grid=(2,), in_specs=(spec,), out_specs=spec)

    def function(x):
        y = call(x).reshape(16, 16)
        return jnp.broadcast_to(jnp.concatenate((y[8:], y[:8])), (2, 16, 16))

    graph = jax.make_jaxpr(function)(signature)
    result = compare_pallas_kernels(graph, graph, equivalence="semantic", output_boundary="parent")
    x = np.arange(256, dtype=np.float32)
    root = result.left_protocol.projection.outputs[0]
    for point in ((0, 0, 0), (1, 15, 15), (1, 8, 4)):
        _, i, j = point
        assert at_point(root, (x,), point) == x[((i + 8) % 16) * 16 + j]


def test_parent_function_comparison_keeps_output_pytree():
    function, signature = discarded_function()
    with pytest.raises(ComparisonContractError, match="output_tree"):
        compare_functions(
            function,
            lambda x: (function(x),),
            signature,
            scope="device_local",
            output_boundary="parent",
        )


def test_parent_shard_output_assembly_remains_explicitly_unsupported():
    function, signature = discarded_function()
    shard = jax.shard_map(
        function,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:1]), ("d",)),
        in_specs=PartitionSpec(),
        out_specs=PartitionSpec(),
        check_vma=False,
    )
    graph = jax.make_jaxpr(shard)(signature)
    with pytest.raises(UnsupportedJaxprError, match="shard/core assembly"):
        compare_pallas_kernels(graph, graph, output_boundary="parent")


@pytest.mark.parametrize("mode", ["out_of_bounds", "in_flight", "nested_consumer"])
def test_unused_read_exception_preserves_address_protocol_and_nested_consumption(mode):
    signature = jax.ShapeDtypeStruct((128,), jnp.float32)

    def kernel(x, y, temporary, semaphore):
        if mode == "out_of_bounds":
            loaded = pp.load(temporary, (jnp.arange(128) + 128,))
            y[...] = loaded
        elif mode == "in_flight":
            transfer = tpu.async_copy(x, temporary, semaphore)
            y[...] = temporary[...]
            transfer.wait()
        else:
            loaded = temporary[...]

            def consume(ref):
                ref[...] = loaded * 0
                y[...] = ref[...]

            pl.run_scoped(consume, tpu.VMEM((128,), jnp.float32))
        y[...] = x[...]

    graph = jax.make_jaxpr(
        pl.pallas_call(
            kernel,
            out_shape=signature,
            scratch_shapes=(tpu.VMEM((128,), jnp.float32), tpu.SemaphoreType.DMA),
        )
    )(signature)
    if mode != "nested_consumer":
        # Tracing DCE removes unused loads. Retain the source read explicitly,
        # replacing only its one value-consuming store with a dropped result.
        outer = graph.jaxpr.eqns[0]
        kernel = outer.params["jaxpr"]
        load = next(
            eq
            for eq in kernel.eqns
            if eq.primitive.name in ("get", "masked_load") and eq.invars[0] is kernel.invars[2]
        )
        old = load.outvars[0]
        eqns = [
            eq.replace(outvars=[core.DropVar(old.aval)]) if eq is load else eq
            for eq in kernel.eqns
            if old not in eq.invars
        ]
        assert len(eqns) == len(kernel.eqns) - 1
        kernel = kernel.replace(eqns=eqns)
        graph = graph.replace(
            jaxpr=graph.jaxpr.replace(
                eqns=[outer.replace(params={**outer.params, "jaxpr": kernel})]
            )
        )
    with pytest.raises(
        UnsupportedJaxprError, match="bounds|outside|identity lane|DMA|flight|uninitialized"
    ):
        compare_pallas_kernels(graph, graph, output_boundary="parent")
