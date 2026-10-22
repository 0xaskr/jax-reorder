"""The public local contract reaches proven values and mathematical comparison."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    ComparisonContractError,
    CorrespondenceError,
    Status,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)


def function(*, block=128, mode="add", shard=False, device=False, repeated=False):
    def kernel(x, y):
        value = x[...]
        if repeated:
            y[...] = value - 7
        result = value + value if mode == "add" else value * (3 if mode == "wrong" else 2)
        if device:
            d = lax.axis_index("d").astype(jnp.float32)
            result += (1 - d) if mode == "wrong_device" else d
        y[...] = result

    spec = pl.BlockSpec((block,), lambda p: (p,))
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        grid=(256 // block,),
        in_specs=[spec],
        out_specs=spec,
    )
    if shard:
        call = jax.shard_map(
            call,
            mesh=Mesh(np.asarray(jax.devices()[:2]), ("d",)),
            in_specs=P("d"),
            out_specs=P("d"),
            check_vma=False,
        )
    return call


@pytest.mark.parametrize("shard", [False, True])
def test_retile_and_algebra_are_distinct_local_comparison_stages(shard):
    sig = jax.ShapeDtypeStruct((512 if shard else 256,), jnp.float32)
    left = jax.make_jaxpr(function(shard=shard))(sig)
    retiled = jax.make_jaxpr(function(block=256, shard=shard))(sig)
    graph = compare_pallas_kernels(left, retiled, equivalence="semantic")
    assert graph.status is Status.EQUIVALENT and graph.equivalence_kind == "normalized"
    assert graph.protocol_checked and graph.scope == "device_local"
    verify_pallas_kernel_comparison(left, retiled, graph)
    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(
            left, retiled, replace(graph, evidence=replace(graph.evidence, kind="structural"))
        )
    for mode in ("mul", "wrong"):
        right = jax.make_jaxpr(function(shard=shard, mode=mode))(sig)
        assert (
            compare_pallas_kernels(left, right, equivalence="semantic").status is Status.DIFFERENT
        )
        result = compare_algebraic_jaxprs(left, right, scope="device_local")
        assert result.scope == "device_local"
        assert result.status is (Status.EQUIVALENT if mode == "mul" else Status.INEQUIVALENT)
        assert result.backend_result is not None and result.normalizations


def test_one_dimensional_sequential_full_stores_use_latest_version():
    sig = jax.ShapeDtypeStruct((256,), jnp.float32)
    left = jax.make_jaxpr(function(repeated=True))(sig)
    for mode in ("mul", "wrong"):
        right = jax.make_jaxpr(function(mode=mode))(sig)
        for scope in ("function", "device_local"):
            result = compare_algebraic_jaxprs(left, right, scope=scope)
            assert result.status is (Status.EQUIVALENT if mode == "mul" else Status.INEQUIVALENT)


def test_function_local_scope_excludes_parent_outputs_but_keeps_input_binding():
    sig = jax.ShapeDtypeStruct((256,), jnp.float32)
    call = function()

    def left(x):
        return {"kernel": call(x), "outer": x}

    def right(x):
        return function(mode="mul")(x) + 9

    with pytest.raises(ComparisonContractError):
        compare_functions(left, right, sig)
    result = compare_functions(left, right, sig, scope="device_local")
    assert result.status is Status.EQUIVALENT and result.scope == "device_local"
    assert result.modes[0].scope == "device_local"
    a = jax.make_jaxpr(lambda x, y: call(x))(sig, sig)
    b = jax.make_jaxpr(lambda x, y: call(y))(sig, sig)
    with pytest.raises(ComparisonContractError, match="local"):
        compare_algebraic_jaxprs(a, b, scope="device_local")


def test_device_dependent_math_uses_only_the_bound_device_range():
    sig = jax.ShapeDtypeStruct((512,), jnp.float32)
    left = jax.make_jaxpr(function(shard=True, device=True))(sig)
    for mode in ("mul", "wrong_device"):
        right = jax.make_jaxpr(function(shard=True, device=True, mode=mode))(sig)
        result = compare_algebraic_jaxprs(left, right, scope="device_local")
        assert result.status is (Status.EQUIVALENT if mode == "mul" else Status.INEQUIVALENT)


def test_local_values_keep_peer_identity_without_requiring_global_replica_equality():
    from tests.pallas_remote_mask_cases import masked_function

    sig = jax.ShapeDtypeStruct((16, 256), jnp.float32)
    a = jax.make_jaxpr(masked_function("data_store", replicated_output=True))(sig)
    b = jax.make_jaxpr(masked_function("conditional_store", replicated_output=True))(sig)
    result = compare_algebraic_jaxprs(a, b, scope="device_local")
    assert result.status is Status.EQUIVALENT
    assert result.normalizations[0].remote.local_outputs == ((0, 0), (0, 1))
    wrong = jax.make_jaxpr(masked_function("data_store", wrong_peer=True, replicated_output=True))(
        sig
    )
    assert compare_algebraic_jaxprs(a, wrong, scope="device_local").status is Status.INEQUIVALENT


@pytest.mark.parametrize(
    "options", [{"scope": "all"}, {"scope": "device_local", "modes": ("jvp",)}]
)
def test_invalid_local_function_scope_fails_at_the_entry(options):
    with pytest.raises(ValueError):
        compare_functions(
            function(), function(), jax.ShapeDtypeStruct((256,), jnp.float32), **options
        )


def remote_bound_inputs(mode):
    """Keep wrapper inputs/outputs separate from the remote call's argument order."""
    from jax.experimental.pallas import tpu

    def kernel(a, b, y, send, recv):
        tpu.async_remote_copy(a, y, send, recv, (1 - lax.axis_index("d"),)).wait()
        y[:] = y[:] + b[:]

    spec = pl.BlockSpec((128,), lambda _: (0,))
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec, spec),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.DMA,) * 2,
        compiler_params=tpu.CompilerParams(),
    )

    @jax.shard_map(
        mesh=Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)),
        in_specs=(P("d"), P()),
        out_specs=P("d"),
        check_vma=False,
    )
    def outer(x, z):
        if mode == "before":
            return x, call(z, x)
        if mode == "after":
            return call(z, x), x
        assert mode == "duplicate"
        return call(x, x)

    return jax.make_jaxpr(outer)(
        jax.ShapeDtypeStruct((256,), jnp.float32),
        jax.ShapeDtypeStruct((128,), jnp.float32),
    )


@pytest.mark.parametrize("mode", ["before", "after", "duplicate"])
def test_remote_call_inputs_keep_their_actual_shard_placement(mode):
    from tests.pallas_remote_cases import projection_array

    source = remote_bound_inputs(mode)
    semantic = compare_pallas_kernels(source, source, equivalence="semantic")
    assert semantic.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, semantic)
    result = compare_algebraic_jaxprs(source, source, scope="device_local")
    assert result.status is Status.EQUIVALENT
    proof = result.normalizations[0]
    x, z = np.arange(256, dtype=np.float32), np.arange(128, dtype=np.float32) * 10
    inputs = (x,) if mode == "duplicate" else (x, z)
    for device, output in enumerate(proof.projection.outputs):
        own = x[device * 128 : (device + 1) * 128]
        peer = x[(1 - device) * 128 : (2 - device) * 128] if mode == "duplicate" else z
        np.testing.assert_array_equal(projection_array(output, inputs, 128), own + peer)
    if mode == "before":
        assert (
            compare_pallas_kernels(
                source, remote_bound_inputs("after"), equivalence="semantic"
            ).status
            is Status.EQUIVALENT
        )


def test_remote_source_shape_is_not_erased_by_the_local_structure_shortcut():
    from tests.pallas_remote_cases import remote_graph

    partitioned = remote_graph(in_spec=P("d"))
    replicated = remote_graph(in_spec=P())
    result = compare_pallas_kernels(partitioned, replicated, equivalence="semantic")
    assert result.status is Status.DIFFERENT
    assert result.left_protocol.boundaries != result.right_protocol.boundaries
    with pytest.raises(ComparisonContractError, match="local"):
        compare_algebraic_jaxprs(partitioned, replicated, scope="device_local")


@pytest.mark.parametrize("mathematical", [False, True])
def test_explicit_local_core_proof_survives_function_and_math_entries(mathematical):
    from jax_reorder import CoreExecutionBinding, UnsupportedJaxprError
    from tests.pallas_scheduling_cases import routed

    call = routed()
    sig = jax.ShapeDtypeStruct((8, 256), jnp.float32)
    binding = CoreExecutionBinding((((0,),), ((1,),)))

    def left(x):
        return call(x + x)

    def right(x):
        return call(x * 2)

    result = (
        compare_functions(
            left, right if mathematical else left, sig, scope="device_local", execution=binding
        )
        .modes[0]
        .algebra
    )
    assert result is not None and result.status is Status.EQUIVALENT
    assert result.equivalence_kind == ("algebraic" if mathematical else "structural")
    assert len(result.core_executions) == 2
    for core in result.core_executions:
        assert core is not None and core.binding == binding
        assert len(core.protocol.counters) == 1
        assert {(p.core, p.point) for p in core.programs} == {(0, (0,)), (1, (1,))}
        assert set(core.protocol.assumptions) <= set(result.assumptions)
    swapped = CoreExecutionBinding((((1,),), ((0,),)))
    with pytest.raises(UnsupportedJaxprError, match="no consumer|conserved"):
        compare_functions(left, left, sig, scope="device_local", execution=(binding, swapped))


@pytest.mark.parametrize("mode", ["add", "mul", "wrong"])
def test_unused_manual_mesh_does_not_change_the_bound_local_values(mode):
    left, right = function(), function(shard=True, mode=mode)
    local = jax.ShapeDtypeStruct((256,), jnp.float32)
    global_ = jax.ShapeDtypeStruct((512,), jnp.float32)
    a, b = jax.make_jaxpr(left)(local), jax.make_jaxpr(right)(global_)
    structural = compare_pallas_kernels(a, b)
    assert structural.status is (Status.EQUIVALENT if mode == "add" else Status.DIFFERENT)
    semantic = compare_pallas_kernels(a, b, equivalence="semantic")
    assert semantic.status is (Status.EQUIVALENT if mode == "add" else Status.DIFFERENT)
    mathematical = compare_algebraic_jaxprs(a, b, scope="device_local")
    expected = Status.INEQUIVALENT if mode == "wrong" else Status.EQUIVALENT
    assert mathematical.status is expected
    assert (
        compare_functions(left, right, local, right_args=(global_,), scope="device_local").status
        is expected
    )
    with pytest.raises(ComparisonContractError):
        compare_algebraic_jaxprs(a, b)
    if mode == "add":
        verify_pallas_kernel_comparison(a, b, semantic)
        original = semantic.evidence.right
        assert original.wrappers[0].shard_map.mesh.axes == (("d", 2, "Auto"),)
        assert original.arguments[0].operands[0].type.shape == (512,)
        assert semantic.right_protocol.boundaries[0].inputs[0].shape == (256,)
        wrapper = original.wrappers[0]
        changed = replace(
            wrapper,
            shard_map=replace(
                wrapper.shard_map, mesh=replace(wrapper.shard_map.mesh, axes=(("d", 1, "Auto"),))
            ),
        )
        forged = replace(
            semantic,
            evidence=replace(semantic.evidence, right=replace(original, wrappers=(changed,))),
        )
        with pytest.raises(CorrespondenceError, match="bindings do not replay"):
            verify_pallas_kernel_comparison(a, b, forged)


def test_local_mesh_projection_keeps_used_axes_and_their_finite_ranges():
    local_call = function(device=True)
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]).reshape((2, 1)), ("d", "unused"))
    right = jax.shard_map(local_call, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)
    sig = jax.ShapeDtypeStruct((512,), jnp.float32)
    a = jax.make_jaxpr(function(shard=True, device=True))(sig)
    b = jax.make_jaxpr(right)(sig)
    result = compare_pallas_kernels(a, b, equivalence="semantic")
    assert result.status is Status.EQUIVALENT
    assert result.evidence.axes == (("d", 2, "Auto"),)
    verify_pallas_kernel_comparison(a, b, result)
    smaller = jax.shard_map(
        local_call,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:1]), ("d",)),
        in_specs=P("d"),
        out_specs=P("d"),
        check_vma=False,
    )
    c = jax.make_jaxpr(smaller)(jax.ShapeDtypeStruct((256,), jnp.float32))
    assert compare_pallas_kernels(a, c, equivalence="semantic").status is Status.DIFFERENT
    with pytest.raises(ComparisonContractError, match="local"):
        compare_algebraic_jaxprs(a, c, scope="device_local")


def test_local_mesh_projection_preserves_other_context_and_array_facts():
    from jax_reorder.core.adapters.jax import adapt
    from jax_reorder.core.call_bindings import extract_call
    from jax_reorder.core.index_domain import AnalysisBudget
    from jax_reorder.core.local import _key, _local_facts

    source = jax.make_jaxpr(function(shard=True))(jax.ShapeDtypeStruct((512,), jnp.float32))
    site = extract_call(adapt(source).view, budget=AnalysisBudget())
    local = _local_facts(site)
    typ = site.arguments[0].type
    assert local(typ).shape == typ.shape and not local(typ).sharding.mesh.axes
    for changed in (
        replace(typ, dtype="float16"),
        replace(typ, weak_type=not typ.weak_type),
        replace(typ, metadata=typ.metadata + (("layout_review_marker", 1),)),
        replace(typ, sharding=replace(typ.sharding, memory_kind="reviewed_other_memory")),
        replace(typ, sharding=replace(typ.sharding, partitions=(("d",),))),
    ):
        assert local(typ) != local(changed)
    tag, values = site.operation.context
    context = (
        tag,
        tuple(
            (name, (value[0], not value[1]) if name == "threefry_partitionable" else value)
            for name, value in values
        ),
    )
    altered = replace(site, operation=replace(site.operation, context=context))
    assert _key(site) != _key(altered)
    assert local(site.operation.context) != local(context)
