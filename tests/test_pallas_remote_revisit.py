"""Remote output revisits preserve only ordered versions and real initialization."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from jax.experimental.pallas import tpu

from jax_reorder import (
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_remote_cases import projection_array, remote_graph
from tests.pallas_remote_revisit_cases import WIDTH, expected, graph


def normalized(**options):
    return normalize_view(adapt(graph(**options)).view, budget=AnalysisBudget(100000))


def test_local_transformed_dma_is_not_ignored_inside_a_remote_protocol():
    def kernel(x, y, temporary, local, send, receive):
        tpu.async_copy(
            x.reshape((2, 128)).transpose((1, 0)), temporary.reshape((128, 2)), local
        ).wait()
        peer = 1 - jax.lax.axis_index("d")
        tpu.async_remote_copy(temporary, y, send, receive, (peer,)).wait()

    source = remote_graph(kernel, arrays=1, semaphores=3, size=256)
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
    x = np.arange(512, dtype=np.float32)
    expected = x.reshape(2, 2, 128)[::-1].transpose(0, 2, 1).reshape(-1)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 512), expected
    )


def test_transformed_local_wait_cannot_consume_a_larger_remote_receive():
    def kernel(x, y, send, receive):
        peer = 1 - jax.lax.axis_index("d")
        transfer = tpu.async_remote_copy(x, y, send, receive, (peer,))
        transfer.wait_send()
        tpu.make_async_copy(
            x.reshape((2, 128)).at[0, :], y.reshape((2, 128)).at[0, :], receive
        ).wait()

    source = remote_graph(kernel, size=256)
    with pytest.raises(UnsupportedJaxprError, match="semaphore count is not conserved"):
        normalize_view(adapt(source).view, budget=AnalysisBudget(100000))


def test_ordered_remote_revisit_keeps_final_a_and_previous_b():
    proof = normalized()
    x = np.arange(6 * WIDTH, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 4 * WIDTH), expected(x)
    )
    io = proof.remote.protocol.io
    output = io.ports[1]
    assert output.offsets == (0, 0, 0)
    for device in range(2):
        versions = [
            v
            for v in proof.remote.versions
            if v.storage.ref == output.hbm and v.storage.device == device
        ]
        assert [v.span.start for v in versions] == [0, WIDTH, 0]
        assert [v.version for v in versions] == [1, 2, 3]
        transfers = [
            t
            for t in proof.remote.transfers
            if proof.remote.protocol.protocol.events[t.start].device == device
        ]
        assert len(transfers) == 3
        assert proof.remote.protocol.protocol.precedes(
            transfers[0].receive_complete, transfers[2].start
        )


def test_wait_order_variants_replay_the_same_revisit_proof():
    a, b = (adapt(graph(receive_first=first)).view for first in (False, True))
    result = compare_program_views(a, b, analysis_budget=100000)
    assert result.status is Status.EQUIVALENT
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("mode", ["normal", "wrong_block", "stale_accumulate", "partial_revisit"])
def test_automatic_revisit_requires_a_source_writeback_order_contract(mode):
    with pytest.raises(UnsupportedJaxprError, match="source writeback-order contract"):
        normalized(explicit=False, mode=mode)


@pytest.mark.parametrize("mode", ["missing_send", "missing_receive", "no_ack"])
def test_revisit_never_supplies_missing_completion_or_peer_dependencies(mode):
    with pytest.raises(UnsupportedJaxprError):
        normalized(mode=mode)


@pytest.mark.parametrize("mode", ["stale_accumulate", "partial_revisit"])
def test_changing_automatic_window_does_not_restore_old_staging_values(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalized(explicit=False, mode=mode, revisit=False)


def test_explicit_hbm_partial_revisit_keeps_untouched_old_values():
    proof = normalized(explicit=True, mode="partial_revisit")
    x = np.arange(6 * WIDTH, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 4 * WIDTH),
        expected(x, mode="partial_revisit"),
    )


@pytest.mark.parametrize("mode", ["wrong_block", "self_target"])
def test_wrong_revisited_window_or_peer_is_a_real_output_difference(mode):
    a, b = (adapt(graph(mode=m)).view for m in ("normal", mode))
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(a, b, analysis_budget=100000)
    x = np.arange(6 * WIDTH, dtype=np.float32)
    outputs = [
        projection_array(
            normalize_view(v, budget=AnalysisBudget(100000)).projection.outputs[0], (x,), 4 * WIDTH
        )
        for v in (a, b)
    ]
    np.testing.assert_array_equal(outputs[0], expected(x))
    np.testing.assert_array_equal(outputs[1], expected(x, mode=mode))
    assert not np.array_equal(*outputs)


def test_repeated_hbm_overwrite_does_not_cover_an_unwritten_other_block():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalized(mode="missing_block")


def test_replay_rejects_forged_revisit_hbm_version():
    a, b = (adapt(graph(receive_first=first)).view for first in (False, True))
    result = compare_program_views(a, b, analysis_budget=100000)
    left = result.evidence.left
    remote = left.remote
    output = remote.protocol.io.ports[1].hbm
    versions = list(remote.versions)
    target = next(i for i, v in enumerate(versions) if v.storage.ref == output and v.version == 3)
    versions[target] = replace(versions[target], version=1)
    forged = replace(
        result.evidence, left=replace(left, remote=replace(remote, versions=tuple(versions)))
    )
    with pytest.raises(CorrespondenceError):
        verify_normalized_evidence(a, b, forged, budget=AnalysisBudget(100000))


def test_revisited_output_and_fixed_accumulator_keep_independent_lifetimes():
    proof = normalized(multiple=True)
    x = np.arange(6 * WIDTH, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 4 * WIDTH), expected(x)
    )
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[1], (x,), 2 * WIDTH),
        x.reshape(2, 3, WIDTH).sum(axis=1).reshape(-1),
    )
    total = proof.remote.protocol.io.ports[2]
    assert total.offsets == (0, 0, 0)
    assert len([v for v in proof.remote.versions if v.storage.ref == total.hbm]) == 2


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["automatic", "explicit", "partial", "wrong", "multi"])
@pytest.mark.parametrize("mode", ["on_wait", "eager"])
def test_official_cpu_interpreter_revisit_boundaries(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_remote_revisit_cases",
        case,
        mode,
        entry="run_cpu",
        expected=dict(
            case=case,
            mode=mode,
            status="interpreter_rejects_revisit" if case == "automatic" else "ok",
        ),
    )
