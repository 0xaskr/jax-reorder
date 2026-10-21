"""Independent mask propagation and latest-version checks across reviewed layers."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import CorrespondenceError, Status, UnsupportedJaxprError, compare_jaxprs
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget, axis
from jax_reorder.core.memory import _substitute
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_projection_oracle import at_point as _at_point
from tests.pallas_remote_cases import remote_graph


def at_point(root, inputs, point):
    return _at_point(_substitute(root, {"global": axis("output:0")}), inputs, point)


def _normalize(function, shape):
    source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct(shape, jnp.float32))
    return source, normalize_view(adapt(source).view, budget=AnalysisBudget(100000))


def transformed_tail(*, wrong=False, late=False, interpret=False):
    def kernel(x, y):
        raw = x[...].reshape((4, 4)).T
        valid = (jnp.arange(16).reshape((4, 4)).T % 8) < (6 if wrong else 5)
        if late:
            reduced = jnp.sum(raw, axis=1, keepdims=True)
            y[...] = jnp.where(jnp.arange(4)[:, None] < 3, reduced, 7)
        else:
            y[...] = jnp.sum(jnp.where(valid, raw, 7), axis=1, keepdims=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4, 1), jnp.float32),
        in_specs=(pl.BlockSpec((2, 8), lambda: (0, 0)),),
        out_specs=pl.BlockSpec((4, 1), lambda: (0, 0)),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


def test_reshape_transpose_mask_before_reduction_keeps_defined_lane_coordinates():
    source, proof = _normalize(transformed_tail(), (2, 5))
    rng = np.random.default_rng(20261021)
    x = rng.integers(-50, 50, size=(2, 5)).astype(np.float32)
    expected = np.pad(x, ((0, 0), (0, 3)), constant_values=7).reshape(4, 4).T.sum(axis=1)
    sentinel = np.pad(x, ((0, 0), (0, 3)), constant_values=-99999)
    actual = [at_point(proof.projection.outputs[0], (sentinel,), (i, 0)) for i in range(4)]
    np.testing.assert_array_equal(actual, expected)
    assert compare_jaxprs(source, source).status is Status.EQUIVALENT
    view = adapt(source).view
    replay = normalize_view(view, budget=AnalysisBudget(100000))
    verify_normalized_evidence(
        view, view, NormalizedEvidence(proof, replay, ((0, 0),)), budget=AnalysisBudget(100000)
    )


@pytest.mark.parametrize("options", [{"wrong": True}, {"late": True}])
def test_transformed_padding_cannot_escape_through_wrong_or_post_reduction_mask(options):
    with pytest.raises(UnsupportedJaxprError):
        _normalize(transformed_tail(**options), (2, 5))


def no_other_value(*, mode="safe", interpret=False):
    def kernel(x, y):
        mask = jnp.arange(8) % 3 == 1
        loaded = pp.load(x, (slice(None), slice(None)), mask=jnp.broadcast_to(mask, (2, 8)))
        changed = loaded.reshape((4, 4)).T
        selected = jnp.broadcast_to(mask, (2, 8)).reshape((4, 4)).T
        if mode == "wrong_select":
            selected = ~selected
        if mode == "arithmetic_zero":
            y[...] = changed * 0
        elif mode == "undefined_selector":
            y[...] = jnp.where(changed > 0, 1.0, 2.0)
        else:
            y[...] = jnp.where(selected, changed + 3, -7)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4, 4), jnp.float32),
        in_specs=(pl.BlockSpec((2, 8), lambda: (0, 0)),),
        out_specs=pl.BlockSpec((4, 4), lambda: (0, 0)),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


def test_no_other_masked_load_can_be_cleaned_after_coordinate_transforms():
    _, proof = _normalize(no_other_value(), (2, 8))
    x = np.arange(16, dtype=np.float32).reshape(2, 8) - 4
    expected = np.where(np.arange(8) % 3 == 1, x + 3, -7).reshape(4, 4).T
    for point in np.ndindex(expected.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("mode", ["wrong_select", "arithmetic_zero", "undefined_selector"])
def test_no_other_undefined_lanes_are_never_initialized_by_arithmetic_or_selection(mode):
    with pytest.raises(UnsupportedJaxprError):
        _normalize(no_other_value(mode=mode), (2, 8))


def layered_store(*, modulus=3, residue=1, threshold=5, interpret=False):
    def kernel(x, y):
        row = jnp.arange(3)[:, None]
        column = jnp.arange(8)[None, :]
        first = jnp.broadcast_to(column % modulus == residue, (3, 8))
        second = jnp.broadcast_to(column < threshold, (3, 8)) & (row != 1)
        y[...] = x[...]
        pp.store(y, (slice(None), slice(None)), x[...] + 100, mask=first)
        pp.store(y, (slice(None), slice(None)), x[...] + 200, mask=second)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((3, 8), jnp.float32),
        in_specs=(pl.BlockSpec((3, 8), lambda: (0, 0)),),
        out_specs=pl.BlockSpec((3, 8), lambda: (0, 0)),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


@pytest.mark.parametrize("modulus,residue,threshold", [(2, 0, 3), (3, 1, 5), (5, 4, 0), (2, 1, 8)])
def test_overlapping_masked_stores_retain_each_latest_active_lane(modulus, residue, threshold):
    _, proof = _normalize(
        layered_store(modulus=modulus, residue=residue, threshold=threshold), (3, 8)
    )
    rng = np.random.default_rng(99 + threshold)
    x = rng.integers(-100, 100, size=(3, 8)).astype(np.float32)
    first = np.arange(8) % modulus == residue
    second = (np.arange(8) < threshold)[None, :] & (np.arange(3)[:, None] != 1)
    expected = np.where(second, x + 200, np.where(first, x + 100, x))
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


def remote_disjoint_mask(*, overlap=False, missing_wait=False):
    def kernel(x, y, temporary, send, receive):
        peer = 1 - jax.lax.axis_index("d")
        temporary[...] = x[...]
        transfer = tpu.async_remote_copy(temporary.at[:128], y.at[:128], send, receive, (peer,))
        active = jnp.arange(256) >= (127 if overlap else 128)
        pp.store(temporary, (slice(None),), x[...] + 100, mask=active)
        if not missing_wait:
            transfer.wait()
        y[128:] = temporary[128:]

    return kernel


def test_remote_masked_store_disjoint_from_live_dma_uses_exact_active_footprint():
    graph = remote_graph(remote_disjoint_mask(), arrays=1, size=256)
    view = adapt(graph).view
    proof = normalize_view(view, budget=AnalysisBudget(100000))
    x = np.arange(512, dtype=np.float32) - 5
    expected = np.concatenate((x[256:384], x[128:256] + 100, x[:128], x[384:] + 100))
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]
    replay = normalize_view(view, budget=AnalysisBudget(100000))
    assert proof.remote is not None
    version = proof.remote.versions[-1]
    corrupted = replace(
        proof,
        remote=replace(proof.remote, versions=(replace(version, version=999),)),
    )
    with pytest.raises(CorrespondenceError, match="replay"):
        verify_normalized_evidence(
            view,
            view,
            NormalizedEvidence(corrupted, replay, ((0, 0),)),
            budget=AnalysisBudget(100000),
        )


@pytest.mark.parametrize("options", [{"overlap": True}, {"missing_wait": True}])
def test_remote_masked_access_never_removes_conflicts_or_completion_obligations(options):
    graph = remote_graph(remote_disjoint_mask(**options), arrays=1, size=256)
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))


def local_data_mask(*, mode="load_other", interpret=False):
    def kernel(x, y, temporary):
        mask = x[...] > 0
        if mode == "store":
            y[...] = x[...]
            pp.store(y, (slice(None),), x[...] + 9, mask=mask)
            return
        loaded = pp.load(
            temporary if mode == "active_uninitialized" else x,
            (slice(None),),
            mask=mask,
            other=7 if mode in ("load_other", "active_uninitialized") else None,
        )
        if mode in ("cleanup", "wrong_cleanup"):
            loaded = jnp.where(~mask if mode == "wrong_cleanup" else mask, loaded, 7)
        y[...] = loaded

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        in_specs=(pl.BlockSpec((8,), lambda: (0,)),),
        out_specs=pl.BlockSpec((8,), lambda: (0,)),
        scratch_shapes=(tpu.VMEM((8,), jnp.float32),),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


@pytest.mark.parametrize("mode", ["load_other", "store", "cleanup"])
def test_data_mask_is_value_conditional_over_fully_initialized_local_memory(mode):
    _, proof = _normalize(local_data_mask(mode=mode), (8,))
    x = np.asarray([-5, 0, 9, 1, -1, 100, -100, 2], dtype=np.float32)
    expected = np.where(x > 0, x + 9, x) if mode == "store" else np.where(x > 0, x, 7)
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("mode", ["wrong_cleanup", "active_uninitialized"])
def test_data_mask_cannot_initialize_an_unknown_active_local_lane(mode):
    with pytest.raises(UnsupportedJaxprError):
        _normalize(local_data_mask(mode=mode), (8,))


def remote_data_mask(*, mode="load_other"):
    def kernel(x, y, temporary, send, receive):
        peer = 1 - jax.lax.axis_index("d")
        tpu.async_remote_copy(
            x, y if mode == "active_uninitialized" else temporary, send, receive, (peer,)
        ).wait()
        mask = x[...] > 0
        if mode == "store":
            y[...] = temporary[...]
            pp.store(y, (slice(None),), temporary[...] + 9, mask=mask)
            return
        loaded = pp.load(
            temporary,
            (slice(None),),
            mask=mask,
            other=7 if mode in ("load_other", "active_uninitialized") else None,
        )
        if mode in ("cleanup", "wrong_cleanup"):
            loaded = jax.lax.select(
                ~mask if mode == "wrong_cleanup" else mask,
                loaded,
                jnp.full_like(loaded, 7),
            )
        y[...] = loaded

    return kernel


@pytest.mark.parametrize("mode", ["load_other", "store", "cleanup"])
def test_remote_data_mask_uses_local_predicate_with_the_received_version(mode):
    graph = remote_graph(remote_data_mask(mode=mode), arrays=1, size=128)
    proof = normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))
    x = (np.arange(256) % 11 - 4).astype(np.float32)
    received = np.concatenate((x[128:], x[:128]))
    expected = (
        np.where(x > 0, received + 9, received) if mode == "store" else np.where(x > 0, received, 7)
    )
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("mode", ["wrong_cleanup", "active_uninitialized"])
def test_remote_data_mask_cannot_launder_an_unknown_lane_or_mismatched_predicate(mode):
    graph = remote_graph(remote_data_mask(mode=mode), arrays=1, size=128)
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))


def mixed_tail_mask(*, wrong=False):
    def kernel(x, y):
        valid = jnp.arange(8) < 5
        safe = jnp.where(valid, x[...], 0)
        active = (jnp.arange(8) < (6 if wrong else 5)) & (safe >= 0)
        y[...] = pp.load(x, (slice(None),), mask=active, other=7)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        in_specs=(pl.BlockSpec((8,), lambda: (0,)),),
        out_specs=pl.BlockSpec((8,), lambda: (0,)),
        compiler_params=tpu.CompilerParams(),
    )


def test_static_tail_conjunct_restricts_a_data_mask_without_inventing_padding():
    _, proof = _normalize(mixed_tail_mask(), (5,))
    x = np.asarray([-2, 0, 3, -4, 5], dtype=np.float32)
    expected = np.concatenate((np.where(x >= 0, x, 7), np.full(3, 7)))
    for point in np.ndindex((8,)):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


def test_a_data_mask_does_not_hide_a_static_tail_off_by_one():
    with pytest.raises(UnsupportedJaxprError):
        _normalize(mixed_tail_mask(wrong=True), (5,))


def test_shared_data_predicate_does_not_expand_the_expression_dag():
    from jax_reorder.core.masked_access import resolve_defined
    from jax_reorder.core.memory import ScalarType, ValueExpr

    typ = ScalarType("float32", False, ())
    zero = ValueExpr("literal", typ, data=np.float32(0).tobytes())
    undefined = ValueExpr("undefined", typ)
    source = ValueExpr("input", typ, input=0)
    for _ in range(32):
        source = ValueExpr("add", typ, (source, source))
    selector = ValueExpr("gt", ScalarType("bool", False, ()), (source, zero))
    false_branch = ValueExpr("select_n", typ, (selector, zero, undefined))
    true_branch = ValueExpr("select_n", typ, (selector, undefined, zero))
    root = ValueExpr("select_n", typ, (selector, false_branch, true_branch))
    cleaned = resolve_defined(root, {}, AnalysisBudget(1000), "review.shared_data_predicate")
    pending, seen = [cleaned], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, ValueExpr):
            assert value.operation != "undefined"
            pending.extend(value.operands)
    assert len(seen) < 50


def test_large_mixed_mask_work_depends_on_cut_boundaries_not_lanes():
    from jax_reorder.core.index_domain import axis, integer
    from jax_reorder.core.masked_access import mask_regions
    from jax_reorder.core.memory import Predicate, ScalarType, ValueExpr
    from jax_reorder.core.tensor_storage import full_box

    extent = 2**28
    limit = extent - 123
    typ = ScalarType("float32", False, ())
    boolean = ScalarType("bool", False, ())
    data = ValueExpr("input", typ, (axis("local:0"),), input=0)
    nonnegative = ValueExpr("ge", boolean, (data, ValueExpr("literal", typ, data=b"\0" * 4)))
    mask = ValueExpr(
        "and", boolean, (Predicate("lt", (axis("local:0"), integer(limit))), nonnegative)
    )
    regions = mask_regions(
        mask, (full_box((extent,)),), {}, AnalysisBudget(1000), "review.large_mixed_mask"
    )
    assert sum(region.box.size for region in regions) == limit
    assert all(region.box.axes[0].stop <= limit for region in regions)
    assert len(regions) <= 2


def test_shared_boolean_mask_keeps_its_static_envelope_bounded():
    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.masked_access import mask_regions
    from jax_reorder.core.memory import Predicate, ScalarType, ValueExpr
    from jax_reorder.core.tensor_storage import full_box

    boolean = ScalarType("bool", False, ())
    predicate = ValueExpr("input", boolean, (axis("local:0"),), input=0)
    static = Predicate("lt", (axis("local:0"), integer(7)))
    mask = ValueExpr("and", boolean, (static, predicate))
    for _ in range(32):
        mask = ValueExpr("and", boolean, (mask, mask))
    regions = mask_regions(
        mask, (full_box((16,)),), {}, AnalysisBudget(1000), "review.shared_boolean_mask"
    )
    assert sum(region.box.size for region in regions) == 7


def test_undefined_memory_predicate_is_rejected_even_when_the_write_is_overwritten():
    def kernel(x, y):
        partial = pp.load(x, (slice(None),), mask=jnp.arange(8) < 4)
        pp.store(y, (slice(None),), x[...] + 2, mask=partial > 0)
        y[...] = x[...]

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        in_specs=(pl.BlockSpec((8,), lambda: (0,)),),
        out_specs=pl.BlockSpec((8,), lambda: (0,)),
        compiler_params=tpu.CompilerParams(),
    )
    with pytest.raises(UnsupportedJaxprError):
        _normalize(function, (8,))


def test_a_dynamic_mask_can_initialize_and_read_the_same_scratch_lanes():
    def kernel(x, y, temporary):
        mask = x[...] > 0
        pp.store(temporary, (slice(None),), x[...] + 2, mask=mask)
        y[...] = pp.load(temporary, (slice(None),), mask=mask, other=7)

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        in_specs=(pl.BlockSpec((8,), lambda: (0,)),),
        out_specs=pl.BlockSpec((8,), lambda: (0,)),
        scratch_shapes=(tpu.VMEM((8,), jnp.float32),),
        compiler_params=tpu.CompilerParams(),
    )
    _, proof = _normalize(function, (8,))
    x = np.arange(8, dtype=np.float32) - 3
    expected = np.where(x > 0, x + 2, 7)
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


def test_empty_static_conjunct_of_remote_data_mask_keeps_protocol_events_in_sync():
    def kernel(x, y, temporary, send, receive):
        tpu.async_remote_copy(x, temporary, send, receive, (1 - jax.lax.axis_index("d"),)).wait()
        active = (jnp.arange(128) < 0) & (x[...] > 0)
        loaded = pp.load(temporary, (slice(None),), mask=active, other=7)
        y[...] = loaded

    graph = remote_graph(kernel, arrays=1, size=128)
    proof = normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))
    x = np.arange(256, dtype=np.float32) - 100
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == 7
    assert proof.remote is not None
    assert len(proof.remote.transfers) == 2


def consumed_masked_swap(*, remote=False, unsafe=False, data_mask=False):
    width = 128 if remote else 8

    def update(x, y, temporary):
        mask = x[...] > 0 if data_mask else jnp.arange(width) < width // 2
        old = pp.swap(temporary, (slice(None),), x[...] + 9, mask=mask)
        if not unsafe:
            old = jax.lax.select(mask, old, jnp.full_like(old, -7))
        y[...] = old + temporary[...]

    def local_kernel(x, y, temporary):
        temporary[...] = x[...]
        update(x, y, temporary)

    def remote_kernel(x, y, temporary, send, receive):
        tpu.async_remote_copy(x, temporary, send, receive, (1 - jax.lax.axis_index("d"),)).wait()
        update(x, y, temporary)

    if remote:
        return remote_graph(remote_kernel, arrays=1, size=128)
    function = pl.pallas_call(
        local_kernel,
        out_shape=jax.ShapeDtypeStruct((8,), jnp.float32),
        in_specs=(pl.BlockSpec((8,), lambda: (0,)),),
        out_specs=pl.BlockSpec((8,), lambda: (0,)),
        scratch_shapes=(tpu.VMEM((8,), jnp.float32),),
        compiler_params=tpu.CompilerParams(),
    )
    return jax.make_jaxpr(function)(jax.ShapeDtypeStruct((8,), jnp.float32))


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("data_mask", [False, True])
def test_consumed_masked_swap_preserves_active_old_values_and_latest_stored_values(
    remote, data_mask
):
    graph = consumed_masked_swap(remote=remote, data_mask=data_mask)
    proof = normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))
    x = (np.arange(256 if remote else 8) % 11 - 4).astype(np.float32)
    received = np.concatenate((x[128:], x[:128])) if remote else x
    mask = (
        x > 0 if data_mask else np.arange(x.size) % (128 if remote else 8) < (64 if remote else 4)
    )
    expected = np.where(mask, received + x + 9, received - 7)
    for point in np.ndindex(x.shape):
        assert at_point(proof.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("remote", [False, True])
def test_masked_swap_inactive_return_values_remain_undefined(remote):
    graph = consumed_masked_swap(remote=remote, unsafe=True)
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(adapt(graph).view, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("squeeze", [False, True])
def test_dynamic_slice_masks_use_access_result_coordinates_for_each_program(squeeze):
    shape = (2, 8) if squeeze else (8,)

    def kernel(x, y):
        p = pl.program_id(0)
        index = (p, pl.ds(p * 4, 2)) if squeeze else (pl.ds(p * 4, 2),)
        y[...] = pp.load(x, index, mask=jnp.arange(2) < 1, other=7)

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        grid=(2,),
        in_specs=(pl.BlockSpec(shape, lambda p: (0,) * len(shape)),),
        out_specs=pl.BlockSpec((2,), lambda p: (p,)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
    )
    _, proof = _normalize(function, shape)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 10
    expected = [x[0, 0] if squeeze else x[0], 7, x[1, 4] if squeeze else x[4], 7]
    for i, value in enumerate(expected):
        assert at_point(proof.projection.outputs[0], (x,), (i,)) == value


@pytest.mark.parametrize("sanitize", [False, True])
def test_dma_source_definedness_is_required_even_when_the_transfer_result_is_unused(sanitize):
    def kernel(x, y, source, destination, semaphore):
        mask = jnp.arange(256) < 128
        loaded = pp.load(x, (slice(None),), mask=mask)
        source[...] = jnp.where(mask, loaded, 7) if sanitize else loaded
        tpu.async_copy(source, destination, semaphore).wait()
        y[...] = x[...]

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        in_specs=(pl.BlockSpec((256,), lambda: (0,)),),
        out_specs=pl.BlockSpec((256,), lambda: (0,)),
        scratch_shapes=(
            tpu.VMEM((256,), jnp.float32),
            tpu.VMEM((256,), jnp.float32),
            tpu.SemaphoreType.DMA,
        ),
        compiler_params=tpu.CompilerParams(),
    )
    if not sanitize:
        with pytest.raises(UnsupportedJaxprError, match="undefined|uninitialized"):
            _normalize(function, (256,))
    else:
        _, proof = _normalize(function, (256,))
        assert len(proof.projection.communication) == 1
        x = np.arange(256, dtype=np.float32)
        for point in np.ndindex(x.shape):
            assert at_point(proof.projection.outputs[0], (x,), point) == x[point]
