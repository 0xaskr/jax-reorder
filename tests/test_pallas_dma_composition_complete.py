"""Composed Ref addresses preserve actual holes, counts and remote ordering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import normalize_view
from tests.pallas_composed_dma_cases import (
    local_expected,
    local_holes,
    native_padded_view,
    padded_view,
    remote_expected,
    remote_kernel,
)
from tests.pallas_projection_oracle import at_point
from tests.pallas_remote_cases import projection_array, remote_graph


def local_projection(**options):
    source = jax.make_jaxpr(local_holes(**options))(jax.ShapeDtypeStruct((32, 128), jnp.float32))
    return normalize_view(adapt(source).view, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("destination_holes", [False, True])
@pytest.mark.parametrize("mode", ["normal", "source_hole", "partial_source"])
def test_local_holes_keep_exact_values_and_payload_counts(destination_holes, mode):
    proof = local_projection(destination_holes=destination_holes, mode=mode)
    x = np.arange(4096, dtype=np.float32).reshape(32, 128)
    expected = local_expected(x, destination_holes=destination_holes, mode=mode)
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]
    (transfer,) = proof.projection.communication
    assert transfer.transfer.elements == 2048
    assert transfer.transfer.logical_bytes == 8192
    assert len(transfer.transfer.source_region) == 2


def test_local_dma_destination_hole_remains_writable_in_flight():
    proof = local_projection(destination_holes=True, mode="destination_hole")
    x = np.arange(4096, dtype=np.float32).reshape(32, 128)
    expected = local_expected(x, destination_holes=True, mode="destination_hole")
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize(
    "mode",
    [
        "source_overlap",
        "destination_overlap",
        "missing_wait",
        "missing_source",
        "missing_destination",
    ],
)
def test_local_holes_do_not_hide_live_conflicts_or_missing_completion(mode):
    with pytest.raises(UnsupportedJaxprError):
        local_projection(mode=mode, destination_holes=True)


@pytest.mark.parametrize("mode", ["reshape", "transpose", "slice"])
def test_padded_view_preserves_initialized_lane_obligations(mode):
    shape = jax.ShapeDtypeStruct((2, 5), jnp.float32)
    source = jax.make_jaxpr(padded_view(mode=mode))(shape)
    reference = jax.make_jaxpr(
        lambda x: jnp.sum(x[:, 4:] if mode == "slice" else x, axis=1, keepdims=True)
    )(shape)
    result = compare_algebraic_jaxprs(source, reference, algebra_budget=100000)
    assert result.status is Status.EQUIVALENT
    assert any(p and p.deferred_padding for p in result.normalizations)


@pytest.mark.parametrize("mode", ["reshape", "transpose", "slice"])
def test_padded_view_does_not_initialize_extra_lane(mode):
    shape = jax.ShapeDtypeStruct((2, 5), jnp.float32)
    source = jax.make_jaxpr(padded_view(mode=mode, limit=6))(shape)
    reference = jax.make_jaxpr(lambda x: jnp.sum(x, axis=1, keepdims=True))(shape)
    with pytest.raises(MathematicalUndecidedError, match="memory initialization"):
        compare_algebraic_jaxprs(source, reference, algebra_budget=100000)


@pytest.mark.parametrize("mode", ["reshape", "transpose", "mixed", "holes", "indexed"])
@pytest.mark.parametrize("wrong", [False, True])
def test_remote_views_keep_logical_element_order_and_true_footprint(mode, wrong):
    size = 512 if mode == "holes" else 256
    source = remote_graph(remote_kernel(mode=mode, wrong=wrong), arrays=1, semaphores=3, size=size)
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
    x = np.arange(2 * size, dtype=np.float32)
    actual = projection_array(proof.projection.outputs[0], (x,), 2 * size)
    np.testing.assert_array_equal(actual, remote_expected(x, mode=mode, wrong=wrong))
    if mode == "holes":
        starts = [
            c.transfer
            for p in proof.remote.protocol.programs
            for c in p.commands
            if hasattr(c, "transfer")
        ]
        assert all(len(t.source_spans) == len(t.destination_spans) == 2 for t in starts)
        assert all(t.logical_bytes == 1024 for t in starts)


@pytest.mark.parametrize("mode", ["overlap", "missing_wait"])
def test_remote_holes_do_not_remove_conflicts_or_completion_requirements(mode):
    source = remote_graph(remote_kernel(mode=mode), arrays=1, semaphores=3, size=512)
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(adapt(source).view, budget=AnalysisBudget(100000))


def run_tpu_cpu(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_composed_dma_cases",
        case,
        mode,
        entry="run_cpu",
        expected=dict(case=case, mode=mode, status="ok"),
    )


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["on_wait", "eager"])
def test_remote_index_chains_match_official_tpu_cpu(mode):
    run_tpu_cpu("indexed", mode)


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case", ["local", "local_holes", "padding", "reshape", "transpose", "mixed", "holes"]
)
def test_pinned_tpu_interpreter_ref_transform_boundary(case):
    # This is an upstream interpreter rejection, not a numerical validation.
    with pytest.raises(
        AssertionError, match="'ReshapeTransform' object has no attribute 'indices'"
    ):
        run_tpu_cpu(case, "on_wait")


@pytest.mark.pallas_communication
@pytest.mark.parametrize("holes", [False, True])
def test_local_holes_match_generic_official_cpu(holes):
    x = np.arange(4096, dtype=np.float32).reshape(32, 128)
    actual = local_holes(destination_holes=holes, interpret=True)(x)
    np.testing.assert_array_equal(actual, local_expected(x, destination_holes=holes))


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["reshape", "slice"])
def test_padded_views_match_generic_official_cpu(mode):
    x = np.arange(10, dtype=np.float32).reshape(2, 5)
    actual = padded_view(mode=mode, interpret=True)(x)
    np.testing.assert_array_equal(
        actual, np.sum(x[:, 4:] if mode == "slice" else x, axis=1, keepdims=True)
    )


@pytest.mark.pallas_communication
def test_pinned_generic_interpreter_transpose_boundary():
    x = np.arange(10, dtype=np.float32).reshape(2, 5)
    with pytest.raises(NotImplementedError, match="TransposeTransform"):
        padded_view(mode="transpose", interpret=True)(x)


@pytest.mark.pallas_communication
def test_native_padded_view_matches_generic_official_cpu():
    x = np.arange(16 * 384, dtype=np.float32).reshape(16, 384)
    np.testing.assert_array_equal(
        native_padded_view(interpret=True)(x),
        np.broadcast_to(x.sum(axis=1, keepdims=True), (16, 128)),
    )


def test_repeated_exact_footprint_conflict_queries_charge_pair_work():
    from jax_reorder import ComparisonBudgetError
    from jax_reorder.core.communication import LocalDmaState
    from jax_reorder.core.model import DmaView, Origin, ValueRef
    from jax_reorder.core.storage import Span
    from jax_reorder.core.tensor_storage import Box

    occupied = tuple(Box((Span(i, i + 1),)) for i in range(0, 128, 2))
    holes = tuple(Box((Span(i, i + 1),)) for i in range(1, 128, 2))
    dma = DmaView("start", ValueRef(0), ValueRef(1), ValueRef(2), "mesh", priority=0, add=False)
    state = LocalDmaState({2}, AnalysisBudget(4096))
    state.start(
        dma,
        Origin("start"),
        source_version=0,
        source_span=Span(0, 127),
        destination_span=Span(0, 127),
        source_versions=(),
        dtype="float32",
        logical_bytes=256,
        source_region=occupied,
        destination_region=occupied,
    )
    state.access(1, Span(1, 128), write=True, path="holes", region=holes)
    assert state.pending
    with pytest.raises(ComparisonBudgetError, match="dma_overlap"):
        state.access(1, Span(1, 128), write=True, path="holes_again", region=holes)


@pytest.mark.parametrize("consume_early", [False, True])
def test_transformed_partial_remote_wait_preserves_count_dependencies(consume_early):
    from jax.experimental.pallas import tpu

    def kernel(x, y, send, receive):
        transfer = tpu.async_remote_copy(x, y, send, receive, (1 - jax.lax.axis_index("d"),))
        transfer.wait_send()
        tpu.make_async_copy(x.at[:].at[:128], y.at[:].at[:128], receive).wait()
        if consume_early:
            y[:128] = y[:128]
        tpu.make_async_copy(x.at[:].at[128:], y.at[:].at[128:], receive).wait()

    source = remote_graph(kernel, size=256)
    if consume_early:
        with pytest.raises(UnsupportedJaxprError, match="conflict|unordered"):
            normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
    else:
        proof = normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
        x = np.arange(512, dtype=np.float32)
        np.testing.assert_array_equal(
            projection_array(proof.projection.outputs[0], (x,), 512),
            x.reshape(2, 256)[::-1].reshape(-1),
        )


@pytest.mark.parametrize("mode", ["reshape", "transpose", "slice"])
def test_padded_view_wrong_mask_cannot_discard_valid_lanes(mode):
    shape = jax.ShapeDtypeStruct((2, 5), jnp.float32)
    source = jax.make_jaxpr(padded_view(mode=mode, limit=4))(shape)
    reference = jax.make_jaxpr(
        lambda x: jnp.sum(x[:, 4:] if mode == "slice" else x, axis=1, keepdims=True)
    )(shape)
    assert (
        compare_algebraic_jaxprs(source, reference, algebra_budget=100000).status
        is Status.INEQUIVALENT
    )


@pytest.mark.parametrize(
    "source_shape,target_shape,common",
    [
        ((16, 512), (2, 8, 512), 1),
        ((6, 4, 5), (2, 3, 4, 5), 2),
        ((2, 3, 4, 5), (6, 4, 5), 2),
        ((3, 4), (2, 6), 0),
        ((3, 4, 5), (3, 20), 0),
        ((1, 3, 4), (3, 4), 2),
        ((3, 4), (1, 3, 4), 2),
        ((2, 3, 4), (2, 3, 4), 3),
    ],
)
def test_reshape_common_tails_preserve_full_numpy_coordinate_correspondence(
    source_shape, target_shape, common
):
    from jax_reorder.core.index_domain import axis, integer, substitute
    from jax_reorder.core.model import RefTransform
    from jax_reorder.core.ref_views import view_coordinates

    transforms = (RefTransform("reshape", target_shape),)
    shape, backward, forward = view_coordinates(
        source_shape, transforms, "reshape", AnalysisBudget(100000)
    )
    assert shape == target_shape
    source = np.arange(np.prod(source_shape)).reshape(source_shape)
    expected = source.reshape(target_shape)
    for point in np.ndindex(target_shape):
        bindings = {f"local:{i}": integer(v) for i, v in enumerate(point)}
        address = tuple(
            int(substitute(backward[f"local:{i}"], bindings).value)
            for i in range(len(source_shape))
        )
        assert source[address] == expected[point]
    for point in np.ndindex(source_shape):
        bindings = {f"local:{i}": integer(v) for i, v in enumerate(point)}
        address = tuple(
            int(substitute(forward[f"local:{i}"], bindings).value) for i in range(len(target_shape))
        )
        assert expected[address] == source[point]
    for offset in range(1, common + 1):
        assert backward[f"local:{len(source_shape) - offset}"] == axis(
            f"local:{len(target_shape) - offset}"
        )
        assert forward[f"local:{len(target_shape) - offset}"] == axis(
            f"local:{len(source_shape) - offset}"
        )


def test_native_padded_reshape_has_a_complete_public_algebra_proof():
    shape = jax.ShapeDtypeStruct((16, 384), jnp.float32)
    source = jax.make_jaxpr(native_padded_view())(shape)
    reference = jax.make_jaxpr(
        lambda x: jnp.broadcast_to(jnp.sum(x, axis=1, keepdims=True), (16, 128))
    )(shape)
    assert (
        compare_algebraic_jaxprs(source, reference, algebra_budget=100000).status
        is Status.EQUIVALENT
    )
