"""ANY revisits restore only previously completed output versions."""

from dataclasses import replace

import numpy as np
import pytest
from jax.experimental import pallas as pl

from jax_reorder import CorrespondenceError, Status, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.view_graph import match_views
from tests.pallas_buffered_revisit_cases import WIDTH, expected, graph
from tests.pallas_remote_cases import projection_array
from tests.pallas_remote_revisit_cases import graph as explicit_graph


def normalized(**options):
    return normalize_view(adapt(graph(**options)).view, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("mode", ["normal", "partial", "accumulate"])
@pytest.mark.parametrize("programs,consecutive", [(3, 1), (5, 1), (6, 2)])
def test_any_revisit_matches_independent_complete_output(mode, programs, consecutive):
    options = dict(mode=mode, programs=programs, consecutive=consecutive)
    proof = normalized(**options)
    x = np.arange(2 * programs * WIDTH, dtype=np.float32)
    actual = projection_array(proof.projection.outputs[0], (x,), 4 * WIDTH)
    np.testing.assert_array_equal(actual, expected(x, **options))
    if programs == 3 and consecutive == 1 and mode in ("normal", "partial"):
        explicit = normalize_view(
            adapt(explicit_graph(mode="partial_revisit" if mode == "partial" else "normal")).view,
            budget=AnalysisBudget(100000),
        )
        np.testing.assert_array_equal(
            actual, projection_array(explicit.projection.outputs[0], (x,), 4 * WIDTH)
        )
    port = proof.remote.protocol.io.ports[1]
    assert port.restore_on_revisit
    commands = proof.remote.protocol.programs[0].commands
    restored = [command for command in commands if ".restore_read[1]" in command.origin.path]
    assert len(restored) == (programs - 1) // consecutive - 1
    assert not any("program[0].restore" in command.origin.path for command in commands)


def test_revisit_keeps_fixed_output_and_other_port_versions_independent():
    proof = normalized(mode="accumulate", multiple=True)
    x = np.arange(6 * WIDTH, dtype=np.float32)
    for expr, reference in zip(
        proof.projection.outputs, expected(x, mode="accumulate", multiple=True), strict=True
    ):
        np.testing.assert_array_equal(projection_array(expr, (x,), reference.size), reference)
    assert not proof.remote.protocol.io.ports[2].restore_on_revisit


def test_restore_occurs_after_writeback_and_before_remote_destination_write():
    proof = normalized().remote
    protocol = proof.protocol.protocol
    for device in range(2):
        events = [event for event in protocol.events if event.device == device]
        earlier = next(e for e in events if "program[0].output_write[1]" in e.origin.path)
        read = next(e for e in events if "program[2].restore_read[1]" in e.origin.path)
        write = next(e for e in events if "program[2].restore_write[1]" in e.origin.path)
        receive = next(
            e
            for e in protocol.events
            if e.kind == "receive_complete" and "program[2]" in e.origin.path and e.device != device
        )
        assert protocol.precedes(earlier.id, read.id)
        assert protocol.precedes(read.id, write.id)
        assert protocol.precedes(write.id, receive.id)


@pytest.mark.parametrize("mode", ["initial_read", "partial_first", "missing_init"])
def test_any_never_initializes_a_first_visit(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalized(mode=mode)


@pytest.mark.parametrize("mode", ["missing_send", "missing_receive", "no_ack"])
def test_any_does_not_supply_missing_dma_or_peer_synchronization(mode):
    with pytest.raises(UnsupportedJaxprError):
        normalized(mode=mode)


@pytest.mark.parametrize("pipeline_mode", [None, pl.Buffered(1), "immediate"])
def test_immediate_pipeline_still_rejects_nonconsecutive_revisit(pipeline_mode):
    with pytest.raises(UnsupportedJaxprError, match="source writeback-order contract"):
        normalized(pipeline_mode=pipeline_mode)


@pytest.mark.parametrize(
    "pipeline_mode",
    [
        pl.Buffered(2, revisit=pl.RevisitMode.ANY),
        pl.Buffered(1, use_lookahead=True),
        pl.Buffered(1, prefetched_count=1),
    ],
)
def test_unreviewed_pipeline_policy_is_rejected(pipeline_mode):
    with pytest.raises(UnsupportedJaxprError, match="synchronous single-buffer"):
        adapt(graph(pipeline_mode=pipeline_mode))


def test_pipeline_policy_is_retained_in_source_view_and_structural_correspondence():
    any_view, immediate, default = (
        adapt(graph(pipeline_mode=mode)).view for mode in ("any", "immediate", None)
    )
    assert match_views(any_view, immediate) is None
    assert match_views(any_view, default) is None
    assert match_views(immediate, default) is None


@pytest.mark.parametrize("mode", ["normal", "partial", "accumulate"])
def test_wait_variants_replay_restoration_evidence(mode):
    a, b = (adapt(graph(mode=mode, receive_first=order)).view for order in (False, True))
    result = compare_program_views(a, b, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget(100000))
    remote = result.evidence.left.remote
    io = remote.protocol.io
    changed = replace(io.ports[1], restore_on_revisit=False)
    remote = replace(
        remote, protocol=replace(remote.protocol, io=replace(io, ports=(io.ports[0], changed)))
    )
    forged = replace(result.evidence, left=replace(result.evidence.left, remote=remote))
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(a, b, forged, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("mode", ["wrong_block", "self_target"])
def test_wrong_window_or_peer_keeps_a_complete_value_counterexample(mode):
    x = np.arange(6 * WIDTH, dtype=np.float32)
    original, wrong = (normalized(mode=value) for value in ("normal", mode))
    actual = projection_array(wrong.projection.outputs[0], (x,), 4 * WIDTH)
    reference = projection_array(original.projection.outputs[0], (x,), 4 * WIDTH)
    np.testing.assert_array_equal(actual, expected(x, mode=mode))
    assert not np.array_equal(actual, reference)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
def test_official_cpu_interpreter_does_not_implement_any_policy(mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_buffered_revisit_cases",
        mode,
        entry="run_cpu",
        expected=dict(case="buffered_any", mode=mode, status="interpreter_rejects_revisit"),
    )


@pytest.mark.parametrize("executor_name", ["blocked", "local", "tensor"])
@pytest.mark.parametrize("revisit", [None, pl.RevisitMode.IMMEDIATE, pl.RevisitMode.ANY])
def test_local_consumers_accept_disjoint_outputs_without_restoration(executor_name, revisit):
    import jax
    import jax.numpy as jnp

    from jax_reorder.core.local_memory import functionalize_local_call
    from jax_reorder.core.memory import functionalize_call
    from jax_reorder.core.tensor_memory import functionalize_tensor_call

    typ = jax.ShapeDtypeStruct((2 * WIDTH,), jnp.float32)
    input_spec = pl.BlockSpec((WIDTH,), lambda p: (p,))
    output_spec = pl.BlockSpec(
        (WIDTH,), lambda p: (p,), pipeline_mode=pl.Buffered(1, revisit=revisit)
    )

    def kernel(x, y):
        y[:] = x[:] + 1

    source = jax.make_jaxpr(
        pl.pallas_call(
            kernel, out_shape=typ, grid=(2,), in_specs=[input_spec], out_specs=output_spec
        )
    )(typ)
    view = adapt(source).view
    call = view.root.operations[0].call
    executor = {
        "blocked": functionalize_call,
        "local": functionalize_local_call,
        "tensor": functionalize_tensor_call,
    }[executor_name]
    projection = executor(call, budget=AnalysisBudget(100000))
    assert len(projection.outputs) == 1
    normalized = normalize_view(view, budget=AnalysisBudget(100000)).projection
    assert not any("output_restore" in v.origin.path for v in normalized.versions)
