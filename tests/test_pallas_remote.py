"""Remote source facts stay distinct from local DMA and value-equivalence claims."""

from dataclasses import replace

import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from jax_reorder import CorrespondenceError, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import AxisEffectView
from jax_reorder.core.normalization import normalize_view
from jax_reorder.core.remote import analyze_remote_call, verify_remote_protocol
from jax_reorder.core.view_graph import match_views
from tests.pallas_oracle import run_oracle
from tests.pallas_remote_cases import remote_graph, request_reply


def call(view):
    return view.root.operations[0].regions[0].region.operations[0].call


def analyze(view, budget=10000):
    mesh = view.root.operations[0].shard_map.mesh
    return analyze_remote_call(call(view), mesh, budget=AnalysisBudget(budget))


def test_exchange_keeps_mesh_coordinates_and_raw_send_wait_swap():
    view = adapt(remote_graph()).view
    kernel = call(view).kernel
    assert AxisEffectView("d") in kernel.effects
    start, send, receive = (op.dma for op in kernel.operations if op.dma)
    assert start.device_coordinates == send.device_coordinates == receive.device_coordinates
    assert start.source == receive.source == send.destination
    assert start.destination == receive.destination == send.source
    assert start.semaphore == send.source_semaphore == receive.semaphore
    assert start.source_semaphore == send.semaphore == receive.source_semaphore
    assert send.is_wait_send is send.insert_dummy_device is True
    assert receive.is_wait_send is receive.insert_dummy_device is False
    assert match_views(view, view).kind == "structural"
    normalized = normalize_view(view, budget=AnalysisBudget())
    assert normalized.remote is not None
    assert len(normalized.remote.transfers) == 2


def test_conditional_ref_bindings_and_both_device_branches_survive_adaptation():
    view = adapt(remote_graph(request_reply(), semaphores=4)).view
    conditional = call(view).kernel.operations[-1]
    assert conditional.primitive == "cond"
    assert len(conditional.regions) == 2
    assert all(len(b.region.inputs) == len(b.arguments) for b in conditional.regions)
    assert all(any(op.dma for op in b.region.operations) for b in conditional.regions)
    assert match_views(view, view).kind == "structural"
    delayed = adapt(remote_graph(request_reply(delayed=True), semaphores=4)).view
    assert match_views(view, delayed) is None


def test_source_exchange_and_request_reply_have_proven_progress():
    exchange = analyze(adapt(remote_graph()).view)
    assert len(exchange.programs) == 2
    assert len(exchange.protocol.counters) == 4
    assert "default TPU device barrier" in exchange.entry_synchronization
    assert exchange.unproved_obligations
    reply = analyze(adapt(remote_graph(request_reply(), semaphores=4)).view)
    assert len(reply.protocol.counters) == 4
    with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
        analyze(adapt(remote_graph(request_reply(delayed=True), semaphores=4)).view)


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("route", "outside the mesh"),
        ("source", "completion order"),
        ("destination", "completion order"),
    ],
)
def test_source_protocol_rejects_bad_routes_and_inflight_accesses(mode, reason):
    def kernel(x, y, send, receive):
        target = jnp.int32(2) if mode == "route" else 1 - lax.axis_index("d")
        descriptor = pltpu.make_async_remote_copy(x, y, send, receive, (target,))
        descriptor.start()
        if mode == "source":
            x[:] = x[:] + 1
        if mode == "destination":
            y[:] = y[:] + 1
        descriptor.wait()

    with pytest.raises(UnsupportedJaxprError, match=reason):
        analyze(adapt(remote_graph(kernel)).view)


def test_barrier_policy_and_axis_binding_are_not_assumed():
    view = adapt(remote_graph(compiler=pltpu.CompilerParams(skip_device_barrier=True))).view
    with pytest.raises(UnsupportedJaxprError, match="default"):
        analyze(view)
    view = adapt(remote_graph()).view
    mesh = view.root.operations[0].shard_map.mesh
    wrong_axis = replace(mesh, axes=(("other", 2, mesh.axes[0][2]),))
    with pytest.raises(UnsupportedJaxprError, match="device axis differs"):
        analyze_remote_call(call(view), wrong_axis, budget=AnalysisBudget())


def test_source_protocol_evidence_replays_counts_edges_and_platform_assumptions():
    view = adapt(remote_graph()).view
    mesh = view.root.operations[0].shard_map.mesh
    evidence = analyze(view)
    verify_remote_protocol(call(view), mesh, evidence, budget=AnalysisBudget())
    for field, value in (("edges", ()), ("counters", ()), ("assumptions", ())):
        bad = replace(evidence, protocol=replace(evidence.protocol, **{field: value}))
        with pytest.raises(CorrespondenceError, match="does not replay"):
            verify_remote_protocol(call(view), mesh, bad, budget=AnalysisBudget())


def test_constant_remote_target_cannot_fall_through_local_functionalization():
    def kernel(x, y, send, receive):
        pltpu.async_remote_copy(x, y, send, receive, (jnp.int32(0),)).wait()

    view = adapt(remote_graph(kernel)).view
    with pytest.raises(UnsupportedJaxprError, match="count is not conserved|no producer"):
        normalize_view(view, budget=AnalysisBudget())


@pytest.mark.parametrize(
    "mode,reason", [("bounds", "ordered static bounds"), ("overflow", "may overflow")]
)
def test_switch_control_does_not_hide_invalid_bounds_or_intermediate_overflow(mode, reason):
    def kernel(x, y, send, receive):
        device = lax.axis_index("d")
        target = (
            lax.clamp(jnp.int32(2), device, jnp.int32(1))
            if mode == "bounds"
            else lax.clamp(jnp.int32(0), device + jnp.int32(2**31 - 1), jnp.int32(1))
        )
        pltpu.async_remote_copy(x, y, send, receive, (target,)).wait()

    with pytest.raises(UnsupportedJaxprError, match=reason):
        analyze(adapt(remote_graph(kernel)).view)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["exchange", "request_reply"])
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_remote_positive_protocols_against_official_interpreter(case, mode):
    run_oracle(case, mode)


@pytest.mark.pallas_communication
def test_statically_cyclic_protocol_stalls_after_ready_in_isolated_interpreter():
    view = adapt(remote_graph(request_reply(delayed=True), semaphores=4)).view
    with pytest.raises(UnsupportedJaxprError, match="wait dependency cycle"):
        analyze(view)
    # Timeout is only an execution observation; the static graph establishes the cycle.
    with pytest.raises(TimeoutError, match="execution timeout"):
        run_oracle("delayed_request_reply", "on_wait", execution_timeout=5)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["route3_direct", "route3_relay"])
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_three_device_source_protocol_and_official_interpreter(case, mode):
    run_oracle(case, mode, devices=3)
