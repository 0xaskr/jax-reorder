"""Pallas-local replicated reshard is explicit metadata, never global communication."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core, pjit
from jax._src import mesh as mesh_lib
from jax._src.pallas import primitives as pp
from jax.experimental.pallas import tpu
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import UnsupportedJaxprError
from jax_reorder.core.adapters.jax import JaxAdapter, adapt
from jax_reorder.core.adapters.pallas.reshard import adapt_local_reshard
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.local_reshard import validate_local_broadcast, validate_reshard_operation
from jax_reorder.core.model import AdaptationContext, ParameterView
from jax_reorder.core.normalization import normalize_view
from tests.pallas_local_reshard_cases import where_function
from tests.pallas_projection_oracle import at_point
from tests.pallas_remote_cases import remote_graph
from tests.test_pallas_mask_access_integration import at_point as remote_at_point


def graph(**kwargs):
    return jax.make_jaxpr(where_function(**kwargs))(jax.ShapeDtypeStruct((16, 256), jnp.float32))


def reshard_source():
    kernel = graph().eqns[0].params["jaxpr"].eqns[0].params["jaxpr"]
    closed = next(e.params["jaxpr"] for e in kernel.eqns if e.primitive is pjit.jit_p)
    return closed, next(e for e in closed.eqns if e.primitive is pjit.reshard_p)


def typed_reshard(name="reshard"):
    program = adapt(graph()).view
    pending = [program.root]
    while pending:
        region = pending.pop()
        types = {v.id: v.type for v in region.inputs}
        types.update((v.id, v.type) for op in region.operations for v in op.results)
        for op in region.operations:
            if op.primitive == name and (
                name == "reshard"
                or any(p.name == "__pallas_kernel_broadcast" for p in op.parameters)
            ):
                return op, tuple(types[a.id] for a in op.operands)
            pending.extend(b.region for b in op.regions)
            if op.call:
                pending.append(op.call.kernel)
    raise AssertionError("missing reshard")


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("branch", [False, True])
def test_original_jnp_where_retains_local_reshard_and_complete_values(masked, branch):
    program = adapt(graph(masked=masked, branch=branch)).view
    proof = normalize_view(program, budget=AnalysisBudget(100000))
    x = np.arange(4096, dtype=np.float32).reshape(16, 256)
    expected = np.where(np.arange(256)[None, :] < 192, np.concatenate((x[8:], x[:8])), 7)
    actual = np.asarray(
        [at_point(proof.projection.outputs[0], (x,), p) for p in np.ndindex(x.shape)]
    ).reshape(x.shape)
    np.testing.assert_array_equal(actual, expected)
    op, inputs = typed_reshard()
    assert {p.name for p in op.parameters} == {
        "__pallas_kernel_reshard",
        "dst_sharding",
        "concrete_mesh",
    }
    assert inputs[0].sharding != op.results[0].type.sharding
    validate_reshard_operation(op, inputs)


def test_kernel_context_is_inherited_by_jit_and_restored_after_region():
    closed, eq = reshard_source()
    adapter = JaxAdapter(AdaptationContext())
    with mesh_lib.use_abstract_mesh(eq.params["dst_sharding"].mesh):
        adapter.kernel_region(closed, "kernel")
        assert not adapter.in_pallas_kernel
        with pytest.raises(UnsupportedJaxprError, match="only inside a Pallas kernel"):
            adapter.region(closed, "ordinary_jax")
    assert not adapter.in_pallas_kernel
    with pytest.raises(
        UnsupportedJaxprError, match="bound manual mesh|non-default equation context"
    ):
        adapter.kernel_region(closed, "wrong_binding")
    assert not adapter.in_pallas_kernel


@pytest.mark.parametrize(
    "change",
    ["partition", "mesh", "concrete", "parameter", "dtype", "shape", "weak", "source_partition"],
)
def test_reshard_adapter_refuses_unreviewed_contract_changes(change):
    _, eq = reshard_source()
    original = eq.params["dst_sharding"]
    params = dict(eq.params)
    if change == "partition":
        params["dst_sharding"] = original.update(spec=P("d", None))
    elif change == "mesh":
        params["dst_sharding"] = original.update(
            mesh=mesh_lib.AbstractMesh((3,), ("d",), axis_types=(jax.sharding.AxisType.Manual,))
        )
    elif change == "concrete":
        params["concrete_mesh"] = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    elif change == "parameter":
        params["future_layout"] = 1
    elif change == "source_partition":
        partitioned = original.update(
            mesh=mesh_lib.AbstractMesh((2,), ("d",), axis_types=(jax.sharding.AxisType.Explicit,)),
            spec=P("d", None),
        )
        aval = eq.invars[0].aval.update(sharding=partitioned)
        assert aval.sharding.spec == P("d", None)
        eq = eq.replace(invars=[core.Var(aval)])
    else:
        aval = eq.outvars[0].aval.update(
            **{"dtype": jnp.dtype(jnp.int32)}
            if change == "dtype"
            else {"shape": (4, 512)}
            if change == "shape"
            else {"weak_type": True}
        )
        eq = eq.replace(outvars=[core.Var(aval)])
    adapter = JaxAdapter(AdaptationContext())
    adapter.in_pallas_kernel = True
    with mesh_lib.use_abstract_mesh(original.mesh), pytest.raises(UnsupportedJaxprError):
        adapt_local_reshard(eq.replace(params=params), adapter, "reshard")


@pytest.mark.parametrize("change", ["marker", "descriptor", "dtype", "partition", "mesh"])
def test_typed_reshard_identity_rechecks_its_certificate(change):
    op, inputs = typed_reshard()
    if change in ("marker", "descriptor"):
        name = "__pallas_kernel_reshard" if change == "marker" else "dst_sharding"
        op = replace(
            op,
            parameters=tuple(
                ParameterView(p.name, False if p.name == name else p.value) for p in op.parameters
            ),
        )
    else:
        target = op.results[0].type
        target = (
            replace(target, dtype="int32")
            if change == "dtype"
            else replace(target, sharding=replace(target.sharding, partitions=(("d",), ())))
            if change == "partition"
            else replace(
                target,
                sharding=replace(
                    target.sharding, mesh=replace(target.sharding.mesh, axes=(("d", 2, "Auto"),))
                ),
            )
        )
        op = replace(op, results=(replace(op.results[0], type=target),))
    with pytest.raises(UnsupportedJaxprError):
        validate_reshard_operation(op, inputs)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["plain", "branch"])
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_original_where_remote_cpu_execution(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_local_reshard_cases",
        case,
        mode,
        entry="run_cpu",
        expected=dict(case=case, mode=mode, status="ok"),
    )


def test_original_where_public_comparison_and_source_evidence():
    from jax_reorder import (
        Status,
        compare_jaxprs,
        compare_pallas_kernels,
        verify_correspondence,
        verify_pallas_kernel_comparison,
    )

    source = graph()
    exact = compare_jaxprs(source, source)
    assert exact.status is Status.EQUIVALENT
    verify_correspondence(source, source, exact.correspondence)
    assert compare_jaxprs(source, graph(fallback=8)).status is Status.DIFFERENT
    local = compare_pallas_kernels(source, source, check_protocol=True, analysis_budget=100000)
    assert local.status is Status.EQUIVALENT and local.protocol_checked
    verify_pallas_kernel_comparison(source, source, local, analysis_budget=100000)


@pytest.mark.parametrize("descriptor", [None, ("none",)])
def test_explicit_broadcast_cannot_bypass_certificate_with_null_descriptor(descriptor):
    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.tensor_domain import coordinate_ranges
    from jax_reorder.core.tensor_values import tensor_value

    op, inputs = typed_reshard("broadcast_in_dim")
    forged = replace(
        op,
        parameters=tuple(
            ParameterView(p.name, descriptor if p.name == "sharding" else p.value)
            for p in op.parameters
        ),
    )
    with pytest.raises(UnsupportedJaxprError, match="broadcast metadata"):
        tensor_value(
            forged,
            (integer(7),),
            inputs,
            coordinate_ranges("local", op.results[0].type.shape),
            {},
            (1,),
            AnalysisBudget(),
        )


def test_real_where_reshard_cannot_be_promoted_to_an_ordinary_jax_identity():
    closed, eq = reshard_source()
    with mesh_lib.use_abstract_mesh(eq.params["dst_sharding"].mesh):
        with pytest.raises(UnsupportedJaxprError, match="Pallas.kernel.only|only inside a Pallas"):
            adapt(closed)


def test_an_outer_reshard_alone_remains_unsupported():
    closed, eq = reshard_source()
    graph = closed.jaxpr.replace(invars=eq.invars, outvars=eq.outvars, eqns=[eq])
    with mesh_lib.use_abstract_mesh(eq.params["dst_sharding"].mesh):
        with pytest.raises(UnsupportedJaxprError, match="only inside a Pallas kernel"):
            adapt(core.ClosedJaxpr(graph, []))


def test_local_reshard_requires_the_exact_current_manual_mesh_axis_binding():
    _, eq = reshard_source()
    adapter = JaxAdapter(AdaptationContext())
    adapter.in_pallas_kernel = True
    mesh = mesh_lib.AbstractMesh((2,), ("other",), axis_types=(jax.sharding.AxisType.Manual,))
    with (
        mesh_lib.use_abstract_mesh(mesh),
        pytest.raises(UnsupportedJaxprError, match="bound manual mesh"),
    ):
        adapt_local_reshard(eq, adapter, "wrong-axis")


@pytest.mark.parametrize("side", ["source", "target"])
def test_value_manual_axis_metadata_must_be_preserved(side):
    _, eq = reshard_source()
    variable = eq.invars[0] if side == "source" else eq.outvars[0]
    altered = core.Var(
        variable.aval.update(sharding=eq.params["dst_sharding"]).update_manual_axis_type(
            core.ManualAxisType(varying=frozenset({"d"}))
        )
    )
    eq = eq.replace(**{"invars" if side == "source" else "outvars": [altered]})
    adapter = JaxAdapter(AdaptationContext())
    adapter.in_pallas_kernel = True
    with (
        mesh_lib.use_abstract_mesh(eq.params["dst_sharding"].mesh),
        pytest.raises(UnsupportedJaxprError),
    ):
        adapt_local_reshard(eq, adapter, "manual-metadata")


def test_local_reshard_does_not_drop_an_effect():
    _, eq = reshard_source()
    eq = eq.replace(effects=frozenset({core.NamedAxisEffect("d")}))
    adapter = JaxAdapter(AdaptationContext())
    adapter.in_pallas_kernel = True
    with (
        mesh_lib.use_abstract_mesh(eq.params["dst_sharding"].mesh),
        pytest.raises(UnsupportedJaxprError, match="effect schema"),
    ):
        adapt_local_reshard(eq, adapter, "effect")


@pytest.mark.parametrize(
    "change", ["metadata", "memory_kind", "logical_ids", "concrete", "wrong_axis", "auto_axis"]
)
def test_typed_local_reshard_rejects_unreviewed_value_or_mesh_changes(change):
    op, inputs = typed_reshard()
    target = op.results[0].type
    shard = target.sharding
    if change == "metadata":
        target = replace(target, metadata=target.metadata + (("unreviewed_layout", "changed"),))
    elif change == "memory_kind":
        target = replace(target, sharding=replace(shard, memory_kind="different"))
    elif change == "logical_ids":
        target = replace(target, sharding=replace(shard, logical_device_ids=(1, 0)))
    else:
        mesh = (
            replace(shard.mesh, concrete=True)
            if change == "concrete"
            else replace(
                shard.mesh,
                axes=(
                    (
                        "wrong" if change == "wrong_axis" else "d",
                        2,
                        "Auto" if change == "auto_axis" else "Manual",
                    ),
                ),
            )
        )
        target = replace(target, sharding=replace(shard, mesh=mesh))
    op = replace(op, results=(replace(op.results[0], type=target),))
    with pytest.raises(UnsupportedJaxprError):
        validate_reshard_operation(op, inputs)


def test_repeated_certificate_parameter_does_not_hide_behind_dictionary_conversion():
    op, inputs = typed_reshard()
    op = replace(op, parameters=op.parameters + (ParameterView("__pallas_kernel_reshard", True),))
    with pytest.raises(UnsupportedJaxprError, match="metadata does not replay"):
        validate_reshard_operation(op, inputs)


def typed_explicit_broadcast():
    pending = [adapt(graph()).view.root]
    while pending:
        region = pending.pop()
        types = {v.id: v.type for v in region.inputs}
        types.update((v.id, v.type) for op in region.operations for v in op.results)
        for op in region.operations:
            if op.primitive == "broadcast_in_dim" and any(
                p.name == "__pallas_kernel_broadcast" for p in op.parameters
            ):
                return op, tuple(types[a.id] for a in op.operands)
            pending.extend(binding.region for binding in op.regions)
            if op.call:
                pending.append(op.call.kernel)
    raise AssertionError("missing replicated explicit broadcast")


@pytest.mark.parametrize(
    "change", ["metadata", "partition", "concrete", "marker", "duplicate", "shape_parameter"]
)
def test_explicit_local_broadcast_does_not_relax_the_reshard_contract(change):
    op, inputs = typed_explicit_broadcast()
    if change in ("metadata", "partition", "concrete"):
        target = op.results[0].type
        if change == "metadata":
            target = replace(target, metadata=target.metadata + (("unreviewed_layout", "changed"),))
        else:
            shard = target.sharding
            shard = (
                replace(shard, partitions=(("d",), ()))
                if change == "partition"
                else replace(shard, mesh=replace(shard.mesh, concrete=True))
            )
            target = replace(target, sharding=shard)
        op = replace(op, results=(replace(op.results[0], type=target),))
    elif change == "duplicate":
        op = replace(op, parameters=op.parameters + (op.parameters[0],))
    else:
        name = "__pallas_kernel_broadcast" if change == "marker" else "shape"
        op = replace(
            op,
            parameters=tuple(
                ParameterView(p.name, False if change == "marker" else (1,))
                if p.name == name
                else p
                for p in op.parameters
            ),
        )
    with pytest.raises(UnsupportedJaxprError):
        validate_local_broadcast(op, inputs)


@pytest.mark.parametrize("masked", [False, True])
def test_original_where_keeps_local_data_predicate_and_received_peer_values(masked):
    def kernel(x, y, temporary, send, receive):
        tpu.async_remote_copy(x, temporary, send, receive, (1 - jax.lax.axis_index("d"),)).wait()
        mask = x[...] >= 0
        values = pp.load(temporary, (slice(None),), mask=mask) if masked else temporary[...]
        y[...] = jnp.where(mask, values, 7)

    graph = remote_graph(kernel, arrays=1, size=128)
    proof = normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))
    x = (np.arange(256) % 13 - 7).astype(np.float32)
    received = np.concatenate((x[128:], x[:128]))
    expected = np.where(x >= 0, received, 7)
    for point in np.ndindex(x.shape):
        assert remote_at_point(proof.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("axes", [(("d", 3),), (("other", 2),), (("d", 2), ("extra", 1))])
def test_coherent_target_and_parameter_mutation_cannot_change_bound_manual_mesh(axes):
    _, eq = reshard_source()
    bound = eq.params["dst_sharding"].mesh
    mesh = mesh_lib.AbstractMesh(
        tuple(n for _, n in axes),
        tuple(name for name, _ in axes),
        axis_types=(AxisType.Manual,) * len(axes),
    )
    destination = eq.params["dst_sharding"].update(mesh=mesh)
    altered = eq.replace(
        params={**eq.params, "dst_sharding": destination},
        outvars=[core.Var(eq.outvars[0].aval.update(sharding=destination))],
    )
    adapter = JaxAdapter(AdaptationContext())
    adapter.in_pallas_kernel = True
    with (
        mesh_lib.use_abstract_mesh(bound),
        pytest.raises(UnsupportedJaxprError, match="bound manual mesh"),
    ):
        adapt_local_reshard(altered, adapter, "review.coherent_wrong_mesh")


@pytest.mark.parametrize("change", ["metadata", "memory_kind", "logical_device_ids", "devices"])
def test_local_identity_cannot_change_unrelated_array_or_placement_facts(change):
    op, inputs = typed_reshard()
    target = op.results[0].type
    if change == "metadata":
        target = replace(target, metadata=target.metadata + (("unknown_layout", 1),))
    else:
        sharding = target.sharding
        sharding = (
            replace(sharding, memory_kind="VMEM")
            if change == "memory_kind"
            else replace(sharding, logical_device_ids=(0, 1))
            if change == "logical_device_ids"
            else replace(sharding, mesh=replace(sharding.mesh, concrete=True))
        )
        target = replace(target, sharding=sharding)
    changed = replace(op, results=(replace(op.results[0], type=target),))
    with pytest.raises(UnsupportedJaxprError):
        validate_reshard_operation(changed, inputs)


def test_integer_one_is_not_the_explicit_pallas_admission_marker():
    op, inputs = typed_reshard()
    altered = replace(
        op,
        parameters=tuple(
            ParameterView(p.name, 1 if p.name == "__pallas_kernel_reshard" else p.value)
            for p in op.parameters
        ),
    )
    with pytest.raises(UnsupportedJaxprError):
        validate_reshard_operation(altered, inputs)


def test_broadcast_marker_cannot_bypass_revalidation_when_descriptor_is_removed():
    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.tensor_domain import coordinate_ranges
    from jax_reorder.core.tensor_values import tensor_value

    pending = [adapt(graph()).view.root]
    while pending:
        region = pending.pop()
        types = {v.id: v.type for v in region.inputs}
        types.update((v.id, v.type) for op in region.operations for v in op.results)
        for op in region.operations:
            if any(p.name == "__pallas_kernel_broadcast" for p in op.parameters):
                altered = replace(
                    op,
                    parameters=tuple(
                        ParameterView(p.name, None if p.name == "sharding" else p.value)
                        for p in op.parameters
                    ),
                )
                with pytest.raises(UnsupportedJaxprError, match="broadcast metadata"):
                    tensor_value(
                        altered,
                        (integer(7),),
                        tuple(types[v.id] for v in op.operands),
                        coordinate_ranges("local", op.results[0].type.shape),
                        {},
                        (1,),
                        AnalysisBudget(1000),
                    )
                return
            pending.extend(binding.region for binding in op.regions)
            if op.call:
                pending.append(op.call.kernel)
    raise AssertionError("missing explicit broadcast")
