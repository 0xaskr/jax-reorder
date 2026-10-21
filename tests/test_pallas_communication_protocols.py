"""Typed source-to-values checks for regular, reused, partial and mixed protocols."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from jax_reorder import CorrespondenceError, Status, UnsupportedJaxprError, compare_algebraic_jaxprs
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import GlobalEffectView, SemaphoreType
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_protocol_cases import (
    acknowledged_exchange,
    local_regular_function,
    mixed_dma,
    regular_remote,
    shared_local_remote_counter,
    signaled_completion,
)
from tests.pallas_remote_cases import projection_array, remote_graph, sliced_exchange


def normalize(source):
    return normalize_view(adapt(source).view, budget=AnalysisBudget())


def local_graph(kernel=None):
    function = local_regular_function() if kernel is None else local_regular_function(kernel)
    return jax.make_jaxpr(function)(jax.ShapeDtypeStruct((128,), jnp.float32))


def test_regular_types_effects_observations_and_repeated_counter_generations():
    view = adapt(local_graph()).view
    kernel = view.root.operations[0].call.kernel
    assert isinstance(kernel.inputs[-1].type.inner, SemaphoreType)
    assert kernel.inputs[-1].type.inner.kind == "regular"
    assert GlobalEffectView("semaphore") in kernel.effects
    proof = normalize_view(view, budget=AnalysisBudget())
    assert [o.value for o in proof.remote.protocol.protocol.observations] == [3, 0]
    x = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 128), x + 3)
    ordinary = jax.make_jaxpr(lambda x: x + 3)(jax.ShapeDtypeStruct((128,), jnp.float32))
    result = compare_algebraic_jaxprs(local_graph(), ordinary)
    assert result.status is Status.EQUIVALENT
    assert "communication-memory-to-values:v1" in result.rules


@pytest.mark.parametrize("case", ["ack", "mixed", "shared", "partial", "signal", "regular"])
def test_supported_protocol_values_and_public_algebra(case):
    if case == "ack":
        kernel, options = acknowledged_exchange(), dict(size=256, arrays=1, regular=1)

        def expected(x):
            return jnp.concatenate((x[256:], x[:256]))
    elif case == "mixed":
        kernel, options = mixed_dma(), dict(arrays=1, semaphores=3)

        def expected(x):
            return jnp.concatenate((x[128:], x[:128]))
    elif case == "shared":
        kernel, options = shared_local_remote_counter, dict(size=256, arrays=1)

        def expected(x):
            return jnp.concatenate((x[:128], x[384:], x[256:384], x[128:256]))
    elif case == "partial":
        kernel, options = sliced_exchange(split_wait=True), dict(size=256)

        def expected(x):
            return jnp.concatenate((x[384:], x[256:384], x[128:256], x[:128]))
    elif case == "signal":
        kernel, options = signaled_completion(), dict(arrays=1, regular=1)

        def expected(x):
            return jnp.concatenate((x[128:], x[:128]))
    else:
        kernel, options = regular_remote, dict(semaphores=0, regular=1)

        def expected(x):
            return x + 0

    source = remote_graph(kernel, **options)
    proof = normalize(source)
    size = options.get("size", 128) * 2
    x = np.arange(size, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), size), expected(x)
    )
    with jax.set_mesh(source.eqns[0].params["mesh"]):
        ordinary = jax.make_jaxpr(expected)(jax.ShapeDtypeStruct((size,), jnp.float32))
    result = compare_algebraic_jaxprs(source, ordinary)
    assert result.status is Status.EQUIVALENT
    if case in ("mixed", "shared"):
        assert any("valid local DMA" in text for text in result.assumptions)
        assert any("valid remote DMA" in text for text in result.assumptions)


@pytest.mark.parametrize(
    "kernel,options",
    [
        (acknowledged_exchange(early=True), dict(size=256, arrays=1, regular=1)),
        (mixed_dma(early=True), dict(arrays=1, semaphores=3)),
        (signaled_completion(early=True), dict(arrays=1, regular=1)),
    ],
)
def test_unsafe_protocol_changes_fail_before_value_elimination(kernel, options):
    with pytest.raises(UnsupportedJaxprError, match="completion order"):
        normalize(remote_graph(kernel, **options))


def test_counter_reads_are_not_treated_as_pure_ignored_or_constant_values():
    def kernel(x, y, sem):
        target = 1 - jax.lax.axis_index("d")
        pl.semaphore_signal(sem, 1, device_id=target)
        observed = pl.semaphore_read(sem)
        pl.semaphore_wait(sem, 1)
        y[:] = x[:] + observed

    with pytest.raises(UnsupportedJaxprError, match="depends on scheduling"):
        normalize(remote_graph(kernel, semaphores=0, regular=1))


@pytest.mark.parametrize("mutation", ["decrement", "unbalanced", "negative"])
def test_unsupported_counter_policies_are_explicit(mutation):
    def kernel(x, y, sem):
        pl.semaphore_signal(sem, -1 if mutation == "negative" else 1)
        pl.semaphore_wait(
            sem, 0 if mutation == "unbalanced" else 1, decrement=mutation != "decrement"
        )
        y[:] = x[:]

    with pytest.raises(
        UnsupportedJaxprError,
        match={
            "decrement": "decrement=True",
            "unbalanced": "not conserved",
            "negative": "invalid signal",
        }[mutation],
    ):
        normalize(local_graph(kernel))


@pytest.mark.parametrize("field", ["order", "ancestors", "counters", "observations", "producers"])
def test_counter_evidence_and_witness_are_replayed_from_source(field):
    def reordered(x, y, sem):
        pl.semaphore_signal(sem, 3)
        first = pl.semaphore_read(sem)
        pl.semaphore_wait(sem, 3)
        pl.semaphore_signal(sem, 2)
        pl.semaphore_wait(sem, 1)
        pl.semaphore_wait(sem, 1)
        y[:] = x[:] + first + pl.semaphore_read(sem)

    a, b = adapt(local_graph()).view, adapt(local_graph(reordered)).view
    result = compare_program_views(a, b)
    assert result.equivalence_kind == "normalized"
    left = result.evidence.left
    remote = left.remote
    bad_protocol = replace(remote.protocol.protocol, **{field: ()})
    bad = replace(
        result.evidence,
        left=replace(
            left, remote=replace(remote, protocol=replace(remote.protocol, protocol=bad_protocol))
        ),
    )
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(a, b, bad, budget=AnalysisBudget())


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["eager", "on_wait"])
@pytest.mark.parametrize(
    "case",
    [
        "protocol_regular_local",
        "protocol_regular_remote",
        "protocol_ack",
        "protocol_mixed",
        "protocol_shared",
        "protocol_partial",
        "protocol_signal",
    ],
)
def test_protocols_against_official_cpu_interpreter(case, mode):
    from tests.pallas_oracle import run_oracle

    result = run_oracle(case, mode)
    assert result["status"] == "ok" and result["races"] is False


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["protocol_observed_local", "protocol_observed_remote"])
def test_locked_official_interpreter_has_no_semaphore_read_cpu_lowering(case):
    from tests.pallas_oracle import run_oracle

    with pytest.raises(
        AssertionError,
        match="MLIR translation rule for primitive 'semaphore_read' not found for platform cpu",
    ):
        run_oracle(case, "on_wait")
