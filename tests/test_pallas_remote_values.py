"""Remote protocols become complete values only with memory and shard proofs."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    ComparisonBudgetError,
    ComparisonContractError,
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget, integer
from jax_reorder.core.memory import expressions_equal
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_oracle import run_oracle
from tests.pallas_remote_cases import (
    exchange,
    exchange_waits,
    partial_exchange,
    projection_array,
    remote_function,
    remote_graph,
    request_reply,
    reuse_source,
    sliced_exchange,
)


def normalized(kernel=exchange, **options):
    return normalize_view(adapt(remote_graph(kernel, **options)).view, budget=AnalysisBudget())


def boolean_exchange(mode, *, different=False):
    def kernel(x, y, temporary, send, receive):
        device = lax.axis_index("d")
        tpu.async_remote_copy(x, temporary, send, receive, (1 - device,)).wait()
        predicate = device == int(different)
        if mode == "cast":
            added = predicate.astype(jnp.float32)
        elif mode == "via_integer":
            added = predicate.astype(jnp.int32).astype(jnp.float32)
        else:
            added = jnp.where(predicate, jnp.float32(1), jnp.float32(0))
        y[:] = temporary[:] + added

    return kernel


@pytest.mark.parametrize("mode", ["cast", "via_integer"])
@pytest.mark.parametrize("different", [False, True])
def test_remote_boolean_cast_preserves_protocol_and_complete_values(mode, different):
    source = remote_graph(boolean_exchange(mode, different=different), arrays=1)
    reference = remote_graph(boolean_exchange("where"), arrays=1)
    protocol = compare_pallas_kernels(source, reference, check_protocol=True)
    assert protocol.status is Status.DIFFERENT
    assert protocol.left_protocol.remote is not None
    assert protocol.right_protocol.remote is not None
    result = compare_algebraic_jaxprs(source, reference)
    assert result.status is (Status.INEQUIVALENT if different else Status.EQUIVALENT)
    x = np.arange(256, dtype=np.float32)
    exchanged = np.concatenate((x[128:], x[:128]))
    for graph, selected in ((source, int(different)), (reference, 0)):
        proof = normalize_view(adapt(graph).view, budget=AnalysisBudget())
        expected = exchanged + np.repeat(np.arange(2) == selected, 128).astype(np.float32)
        np.testing.assert_array_equal(
            projection_array(proof.projection.outputs[0], (x,), 256), expected
        )


@pytest.mark.parametrize("kind", ["float64_to_float32", "float32_to_bfloat16"])
def test_remote_scalar_float_cast_restores_broadcast_coordinates(kind):
    source_dtype, target_dtype = (
        (jnp.float64, jnp.float32) if kind == "float64_to_float32" else (jnp.float32, jnp.bfloat16)
    )
    expected_scalar = np.asarray(1.003, dtype=source_dtype).astype(target_dtype).astype(np.float32)

    def kernel(cast):
        def body(x, y, temporary, send, receive):
            tpu.async_remote_copy(x, temporary, send, receive, (1 - lax.axis_index("d"),)).wait()
            value = (
                jnp.asarray(1.003, dtype=source_dtype).astype(target_dtype).astype(jnp.float32)
                if cast
                else jnp.float32(expected_scalar)
            )
            y[:] = temporary[:] + value

        return body

    with jax.enable_x64():
        source, reference = (remote_graph(kernel(cast), arrays=1) for cast in (True, False))
        assert compare_algebraic_jaxprs(source, reference).status is Status.EQUIVALENT
        proof = normalize_view(adapt(source).view, budget=AnalysisBudget())
    x = np.arange(256, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 256),
        np.concatenate((x[128:], x[:128])) + expected_scalar,
    )


@pytest.mark.parametrize("route", ["integer", "floating", "data", "outside"])
def test_remote_boolean_conversion_does_not_relax_static_route_requirements(route):
    def kernel(x, y, temporary, send, receive):
        predicate = x[0] > 0 if route == "data" else lax.axis_index("d") == 0
        target = (
            predicate.astype(jnp.float32).astype(jnp.int32)
            if route == "floating"
            else predicate.astype(jnp.int32)
        )
        if route == "outside":
            target += 2
        tpu.async_remote_copy(x, temporary, send, receive, (target,)).wait()
        y[:] = temporary[:]

    source = remote_graph(kernel, arrays=1)
    if route == "integer":
        assert compare_algebraic_jaxprs(source, remote_graph()).status is Status.EQUIVALENT
        assert (
            compare_pallas_kernels(source, source, check_protocol=True).status is Status.EQUIVALENT
        )
    else:
        message = "outside the mesh" if route == "outside" else "static integer"
        for compare in (
            compare_algebraic_jaxprs,
            lambda a, b: compare_pallas_kernels(a, b, check_protocol=True),
        ):
            with pytest.raises(UnsupportedJaxprError, match=message):
                compare(source, remote_graph())


def test_remote_float_boolean_conversion_cannot_supply_semaphore_count():
    def kernel(x, y, temporary, send, receive, semaphore):
        device = lax.axis_index("d")
        tpu.async_remote_copy(x, temporary, send, receive, (1 - device,)).wait()
        count = (device >= 0).astype(jnp.float32).astype(jnp.int32)
        pl.semaphore_signal(semaphore, count)
        pl.semaphore_wait(semaphore, count)
        y[:] = temporary[:]

    source = remote_graph(kernel, arrays=1, regular=1)
    with pytest.raises(UnsupportedJaxprError, match="static integer"):
        compare_pallas_kernels(source, source, check_protocol=True)


def test_wait_reordering_eliminates_schedule_but_retains_remote_versions():
    a, b = (adapt(remote_graph(exchange_waits(receive_first=x))).view for x in (False, True))
    result = compare_program_views(a, b)
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == "normalized"
    assert len(result.evidence.left.remote.transfers) == 2
    remote = result.evidence.left.remote
    assert len(remote.versions) == 6
    assert sum("input_write" in v.origin.path for v in remote.versions) == 2
    assert sum("output_write" in v.origin.path for v in remote.versions) == 2
    assert all(port.ref != port.hbm for port in remote.protocol.io.ports)
    assert len(result.evidence.left.remote.assembly) == 4
    assert result.evidence.left.remote.discharged_obligations
    assert any("valid remote DMA" in text for text in result.assumptions)
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())


def test_request_reply_uses_received_version_as_reply_source():
    proof = normalized(request_reply(), semaphores=4).remote
    first, second = proof.transfers
    assert first.source_versions[0].version == 2  # staging invalidation plus HBM reload
    assert second.source_versions[0].version == 1
    assert proof.protocol.protocol.precedes(first.receive_complete, second.start)


@pytest.mark.parametrize(
    "field", ["versions", "transfers", "assembly", "discharged_obligations", "edges", "value", "io"]
)
def test_replay_rejects_forged_protocol_versions_and_assembly(field):
    a, b = (adapt(remote_graph(exchange_waits(receive_first=x))).view for x in (False, True))
    result = compare_program_views(a, b)
    left = result.evidence.left
    remote = left.remote
    if field == "io":
        io = remote.protocol.io
        altered = replace(io.ports[0], hbm=io.ports[0].ref)
        remote = replace(
            remote,
            protocol=replace(remote.protocol, io=replace(io, ports=(altered, *io.ports[1:]))),
        )
    elif field == "edges":
        remote = replace(
            remote,
            protocol=replace(remote.protocol, protocol=replace(remote.protocol.protocol, edges=())),
        )
    elif field == "value":
        remote = replace(
            remote, versions=(replace(remote.versions[0], value=integer(0)), *remote.versions[1:])
        )
    else:
        remote = replace(remote, **{field: ()})
    bad = replace(result.evidence, left=replace(left, remote=remote))
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(a, b, bad, budget=AnalysisBudget())


def test_scratch_source_can_be_reused_after_send_completion_before_receive_wait():
    proof = normalized(reuse_source(), arrays=1, semaphores=4, size=256)
    assert {v.source_versions[0].version for v in proof.remote.transfers} == {1, 2}
    assert len(proof.remote.transfers) == 4
    assert all(v.storage.device in (0, 1) for v in proof.remote.versions)


@pytest.mark.parametrize("restore", [False, True])
def test_source_overwrite_even_if_restored_cannot_use_a_lucky_topological_order(restore):
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        normalized(reuse_source(early=True, restore=restore), arrays=1, semaphores=4, size=256)


def test_equal_counts_do_not_erase_routes_or_slice_correspondence():
    exchanged = normalized().projection.outputs
    same_device = normalized(exchange_waits(self_target=True)).projection.outputs
    assert not expressions_equal(exchanged, same_device)
    a = normalized(sliced_exchange(), size=256)
    b = normalized(sliced_exchange(swapped=False), size=256)
    assert not expressions_equal(a.projection.outputs, b.projection.outputs)
    assert all(len(c.producers) == 2 for c in a.remote.protocol.protocol.counters)


def test_partial_remote_wait_does_not_select_one_of_the_matching_producers():
    proof = normalized(sliced_exchange(split_wait=True), size=256)
    counters = proof.remote.protocol.protocol.counters
    partial = [c for c in counters if c.consumed_before == 0]
    assert partial and all(not c.required_completions for c in partial)


def test_replication_requires_actual_values_and_supports_replicated_inputs():
    with pytest.raises(UnsupportedJaxprError, match="replicated output values differ"):
        normalized(out_spec=P())
    assert (
        normalized(request_reply(), semaphores=4, out_spec=P()).projection.outputs[0].operation
        == "input"
    )
    assert normalized(in_spec=P(), out_spec=P()).projection.outputs[0].operation == "input"


@pytest.mark.parametrize("mode", ["uninitialized", "gap", "input_write", "send_only"])
def test_unsafe_memory_is_refused_after_protocol_analysis(mode):
    def kernel(x, y, temporary, send, receive):
        target = 1 - lax.axis_index("d")
        if mode == "uninitialized":
            tpu.async_remote_copy(temporary, y, send, receive, (target,)).wait()
        elif mode == "input_write":
            tpu.async_remote_copy(x, x, send, receive, (target,)).wait()
            y[:] = x[:]
        elif mode == "gap":
            tpu.async_remote_copy(x.at[:128], y.at[:128], send, receive, (target,)).wait()
        else:
            copy = tpu.async_remote_copy(x, temporary, send, receive, (target,))
            copy.wait_send()
            if mode != "send_only":
                copy.wait_recv()
            y[:] = temporary[:]
            if mode == "send_only":
                copy.wait_recv()
            else:
                y[:] = x[:]

    with pytest.raises(
        UnsupportedJaxprError, match="uninitialized|read-only|completion order|overlap"
    ):
        normalized(kernel, arrays=1, size=256)


def test_global_shape_and_scratch_lifetime_are_checked_not_declared_away():
    view = adapt(remote_graph()).view
    shard = view.root.operations[0]
    wrong_output = replace(shard.results[0], type=replace(shard.results[0].type, shape=(128,)))
    bad = replace(
        view, root=replace(view.root, operations=(replace(shard, results=(wrong_output,)),))
    )
    with pytest.raises(UnsupportedJaxprError, match="global/local shard"):
        normalize_view(bad, budget=AnalysisBudget())
    source = adapt(remote_graph(reuse_source(), arrays=1, semaphores=4, size=256)).view
    shard = source.root.operations[0]
    branch = shard.regions[0]
    op = branch.region.operations[0]
    changed = replace(
        op.call, scratch=(replace(op.call.scratch[0], lifetime="program"), *op.call.scratch[1:])
    )
    branch = replace(branch, region=replace(branch.region, operations=(replace(op, call=changed),)))
    bad = replace(
        source, root=replace(source.root, operations=(replace(shard, regions=(branch,)),))
    )
    with pytest.raises(UnsupportedJaxprError, match="lifetime"):
        normalize_view(bad, budget=AnalysisBudget())


def test_large_remote_arrays_cost_depends_on_devices_and_boundaries():
    budgets = []
    for size in (128, 2**24):
        budget = AnalysisBudget(512)
        normalize_view(adapt(remote_graph(size=size)).view, budget=budget)
        budgets.append(budget.used)
    assert budgets[0] == budgets[1]
    with pytest.raises(ComparisonBudgetError):
        normalize_view(adapt(remote_graph()).view, budget=AnalysisBudget(20))


@pytest.mark.parametrize("case", ["exchange", "request_reply", "slices", "reuse"])
def test_public_algebra_receives_global_values_and_remote_assumptions(case):
    kernels = {
        "exchange": (exchange, {}, lambda x: jnp.concatenate((x[128:], x[:128]))),
        "request_reply": (
            request_reply(),
            {"semaphores": 4},
            lambda x: jnp.concatenate((x[:128], x[:128])),
        ),
        "slices": (
            sliced_exchange(),
            {"size": 256},
            lambda x: jnp.concatenate((x[384:], x[256:384], x[128:256], x[:128])),
        ),
        "reuse": (
            reuse_source(),
            {"size": 256, "arrays": 1, "semaphores": 4},
            lambda x: jnp.concatenate((x[256:384], x[384:] + 1, x[:128], x[128:256] + 1)),
        ),
    }
    kernel, options, expected = kernels[case]
    a = remote_graph(kernel, **options)
    # Ordinary JAX must produce the same global mesh contract as shard_map.
    with jax.set_mesh(a.eqns[0].params["mesh"]):
        b = jax.make_jaxpr(expected)(
            jax.ShapeDtypeStruct((options.get("size", 128) * 2,), jnp.float32)
        )
    assert compare_jaxprs(a, b).status is Status.DIFFERENT
    result = compare_algebraic_jaxprs(a, b)
    assert result.status is Status.EQUIVALENT
    assert "remote-memory-to-global-values:v2" in result.rules
    assert result.normalizations[0].remote is not None
    assert result.backend_result.queries
    assert any("valid remote DMA" in text for text in result.assumptions)


def test_public_wrong_route_has_complete_output_counterexample():
    result = compare_algebraic_jaxprs(
        remote_graph(), remote_graph(exchange_waits(self_target=True))
    )
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_remote_projection_inside_default_jit():
    a = remote_graph()
    b = jax.make_jaxpr(jax.jit(remote_function(exchange_waits(receive_first=True))))(
        jax.ShapeDtypeStruct((256,), jnp.float32)
    )
    assert compare_algebraic_jaxprs(a, b).status is Status.EQUIVALENT


def test_public_mesh_contract_mismatch_is_not_erased_by_global_projection():
    ordinary = jax.make_jaxpr(lambda x: jnp.concatenate((x[128:], x[:128])))(
        jax.ShapeDtypeStruct((256,), jnp.float32)
    )
    with pytest.raises(ComparisonContractError):
        compare_algebraic_jaxprs(remote_graph(), ordinary)


def test_multiple_inputs_outputs_and_conditional_value_bindings():
    def kernel(reverse):
        def body(x, z, y, w, temporary, s0, r0, s1, r1):
            device = lax.axis_index("d")
            value = lax.cond(device == 0, lambda a, b: a - b, lambda a, b: b - a, x[:], z[:])
            temporary[:] = value
            first = tpu.async_remote_copy(temporary, y, s0, r0, (1 - device,))
            second = tpu.async_remote_copy(z, w, s1, r1, (1 - device,))
            for copy in (second, first) if reverse else (first, second):
                copy.wait()

        return body

    options = dict(inputs=2, outputs=2, arrays=1, semaphores=4)
    a, b = (adapt(remote_graph(kernel(reverse), **options)).view for reverse in (False, True))
    result = compare_program_views(a, b)
    assert result.equivalence_kind == "normalized"
    assert len(result.evidence.left.projection.outputs) == 2
    assert not expressions_equal(*((v,) for v in result.evidence.left.projection.outputs))
    assert {p.port for p in result.evidence.left.remote.assembly if p.role == "input"} == {0, 1}


def test_partial_ordinary_read_write_and_device_data_preserve_coordinates():
    a = remote_graph(partial_exchange, arrays=1, size=256)
    proof = normalize_view(adapt(a).view, budget=AnalysisBudget())
    x = np.arange(512, dtype=np.float32)
    expected = np.concatenate(
        (x[384:] + np.arange(128), x[256:384], x[128:256] + np.arange(128), x[:128] + 1)
    )
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 512), expected
    )
    with jax.set_mesh(a.eqns[0].params["mesh"]):
        b = jax.make_jaxpr(
            lambda x: jnp.concatenate(
                (
                    x[384:] + jnp.arange(128, dtype=jnp.int32).astype(jnp.float32),
                    x[256:384],
                    x[128:256] + jnp.arange(128, dtype=jnp.int32).astype(jnp.float32),
                    x[:128] + 1,
                )
            )
        )(jax.ShapeDtypeStruct((512,), jnp.float32))
    # The existing CAS does not establish the cast conditions over these
    # piecewise coordinate domains; the complete core projection is still valid.
    with pytest.raises(MathematicalUndecidedError, match="conditions"):
        compare_algebraic_jaxprs(a, b)


def test_ordered_remote_write_to_input_is_rejected_even_without_a_race():
    def kernel(x, y, temporary, send, receive):
        temporary[:] = x[:]
        tpu.async_remote_copy(temporary, x, send, receive, (lax.axis_index("d"),)).wait()
        y[:] = x[:]

    with pytest.raises(UnsupportedJaxprError, match="read-only"):
        normalized(kernel, arrays=1)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["remote_slices", "remote_reuse", "remote_partial"])
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_remote_versions_against_official_interpreter(case, mode):
    run_oracle(case, mode)


def test_ordered_staging_output_overwrite_preserves_the_last_value():
    def kernel(x, y, temporary, send, receive):
        tpu.async_remote_copy(x, temporary, send, receive, (1 - lax.axis_index("d"),)).wait()
        y[:] = temporary[:]
        y[:] = x[:]

    proof = normalized(kernel, arrays=1, size=256)
    x = np.arange(512, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 512), x)
    output_buffer = proof.remote.protocol.io.ports[1].ref
    assert sorted(v.version for v in proof.remote.versions if v.storage.ref == output_buffer) == [
        1,
        1,
        2,
        2,
    ]
