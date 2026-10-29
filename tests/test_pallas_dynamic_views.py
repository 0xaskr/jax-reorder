"""Source-bounded addresses and partial Ref aliases retain exact storage state."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import RefTransform, SliceIndex, TensorIndex
from jax_reorder.core.normalization import normalize_view
from jax_reorder.core.ref_views import view_coordinates
from jax_reorder.core.tensor_memory import functionalize_tensor_call
from tests.pallas_pipeline_cases import environment
from tests.pallas_projection_oracle import at_point

PAGE_SHAPE = (2, 8, 128)
PAGE_SIGNATURE = (
    jax.ShapeDtypeStruct((2,), jnp.int32),
    *(jax.ShapeDtypeStruct(PAGE_SHAPE, jnp.float32) for _ in range(2)),
)


def paged_dma_function(
    *, separate_semaphores=False, serialized=False, static_first=False, wrong=None, interpret=False
):
    """K/V page loads preserve the same runtime selections across overlapping DMAs."""

    def kernel(pages, k, v, out_k, out_v, buf_k, buf_v, *sems):
        selected = (jnp.int32(0) if static_first else pages[0] & 1, pages[1] & 1)
        copies = []
        for group, (source, target) in enumerate(((k, buf_k), (v, buf_v))):
            for slot, page in enumerate(selected):
                if wrong == ("k_page" if group == 0 else "v_page") and slot == 1:
                    page = 1 - page
                sem = sems[2 * group + slot] if separate_semaphores else sems[group]
                copy = tpu.make_async_copy(source.at[page, :, :], target.at[slot, :, :], sem)
                copy.start()
                copies.append(copy)
                if serialized:
                    copy.wait()
        if wrong == "destination_hazard":
            buf_k[0, :, :] = jnp.zeros((8, 128), jnp.float32)
        if not serialized:
            for i, copy in enumerate(copies):
                if wrong == "missing_wait" and i == 3:
                    continue
                if wrong == "wrong_wait" and i == 3:
                    tpu.make_async_copy(v.at[selected[1], :, :], buf_v.at[1, :, :], sems[-1]).wait()
                elif wrong == "partial_wait" and i == 3:
                    sem = sems[3] if separate_semaphores else sems[1]
                    tpu.make_async_copy(v.at[selected[1], :4, :], buf_v.at[1, :4, :], sem).wait()
                else:
                    copy.wait()
        out_k[...] = buf_k[...]
        out_v[...] = buf_v[...]

    sem_count = (4 if separate_semaphores else 2) + (wrong == "wrong_wait")
    return pl.pallas_call(
        kernel,
        out_shape=(PAGE_SIGNATURE[1], PAGE_SIGNATURE[2]),
        in_specs=(
            pl.BlockSpec(),
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ),
        scratch_shapes=(
            tpu.VMEM(PAGE_SHAPE, jnp.float32),
            tpu.VMEM(PAGE_SHAPE, jnp.float32),
            *((tpu.SemaphoreType.DMA,) * sem_count),
        ),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


def paged_dma_source(**options):
    return jax.make_jaxpr(paged_dma_function(**options))(*PAGE_SIGNATURE)


def paged_dma_inputs(pages):
    k = np.arange(np.prod(PAGE_SHAPE), dtype=np.float32).reshape(PAGE_SHAPE)
    return np.asarray(pages, np.int32), k, 3 * k + 11


def expected_pages(inputs, *, static_first=False, wrong=None):
    pages, k, v = inputs
    selected = np.bitwise_and(pages, 1).copy()
    if static_first:
        selected[0] = 0
    outputs = []
    for name, source in (("k_page", k), ("v_page", v)):
        indices = selected.copy()
        if wrong == name:
            indices[1] = 1 - indices[1]
        outputs.append(source[indices])
    return tuple(outputs)


@pytest.mark.parametrize(
    "separate_semaphores,static_first", [(False, False), (True, False), (False, True)]
)
def test_bounded_source_dmas_keep_every_page_and_output(separate_semaphores, static_first):
    source = paged_dma_source(separate_semaphores=separate_semaphores, static_first=static_first)
    projection = normalize_view(adapt(source).view, budget=AnalysisBudget()).projection
    assert len(projection.communication) == 4
    for evidence in projection.communication:
        transfer = evidence.transfer
        assert transfer.elements == 8 * 128
        assert transfer.logical_bytes == 8 * 128 * 4
        if transfer.source_alternatives:
            assert len(transfer.source_alternatives) == 2
            assert all(
                sum(box.size for box in choice) == transfer.elements
                for choice in transfer.source_alternatives
            )
    for pages in np.ndindex((2, 2)):
        inputs = paged_dma_inputs(pages)
        expected = expected_pages(inputs, static_first=static_first)
        for output, reference in zip(projection.outputs, expected, strict=True):
            for point in np.ndindex(PAGE_SHAPE):
                assert at_point(output, inputs, point) == reference[point]


@pytest.mark.parametrize("separate_semaphores", [False, True])
def test_bounded_source_dma_overlap_matches_serialized_public_comparison(separate_semaphores):
    concurrent = paged_dma_source(separate_semaphores=separate_semaphores)
    serialized = paged_dma_source(separate_semaphores=separate_semaphores, serialized=True)
    result = compare_pallas_kernels(concurrent, serialized, equivalence="semantic")
    assert result.status is Status.EQUIVALENT
    assert result.protocol_checked
    verify_pallas_kernel_comparison(concurrent, serialized, result)
    assert (
        compare_algebraic_jaxprs(concurrent, serialized, scope="device_local").status
        is Status.EQUIVALENT
    )


@pytest.mark.parametrize("wrong", ["k_page", "v_page"])
def test_bounded_source_dma_wrong_page_keeps_k_and_v_correlations(wrong):
    original, changed = paged_dma_source(), paged_dma_source(wrong=wrong)
    assert (
        compare_pallas_kernels(original, changed, equivalence="semantic").status is Status.DIFFERENT
    )
    projection = normalize_view(adapt(changed).view, budget=AnalysisBudget()).projection
    for pages in np.ndindex((2, 2)):
        inputs = paged_dma_inputs(pages)
        correct, changed_expected = expected_pages(inputs), expected_pages(inputs, wrong=wrong)
        assert not np.array_equal(correct[wrong == "v_page"], changed_expected[wrong == "v_page"])
        for output, reference in zip(projection.outputs, changed_expected, strict=True):
            for point in np.ndindex(PAGE_SHAPE):
                assert at_point(output, inputs, point) == reference[point]
    assert (
        compare_algebraic_jaxprs(original, changed, scope="device_local").status
        is Status.INEQUIVALENT
    )


@pytest.mark.parametrize(
    "wrong", ["destination_hazard", "missing_wait", "partial_wait", "wrong_wait"]
)
def test_bounded_source_dma_does_not_erase_pending_access_or_wait_errors(wrong):
    with pytest.raises(
        UnsupportedJaxprError, match="DMA|dma|uninitialized|completion|pending|wait"
    ):
        normalize_view(adapt(paged_dma_source(wrong=wrong)).view, budget=AnalysisBudget())


def test_bounded_source_dma_cannot_read_an_inflight_destination():
    def kernel(pages, k, v, out_k, out_v, source, destination, first_sem, second_sem):
        first = tpu.async_copy(k, source, first_sem)
        second = tpu.async_copy(source.at[pages[0] & 1, :, :], destination.at[0, :, :], second_sem)
        first.wait()
        second.wait()
        destination[1, :, :] = v[0, :, :]
        out_k[...] = destination[...]
        out_v[...] = destination[...]

    function = pl.pallas_call(
        kernel,
        out_shape=(PAGE_SIGNATURE[1], PAGE_SIGNATURE[2]),
        scratch_shapes=(
            tpu.VMEM(PAGE_SHAPE, jnp.float32),
            tpu.VMEM(PAGE_SHAPE, jnp.float32),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
        ),
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(function)(*PAGE_SIGNATURE)
    with pytest.raises(UnsupportedJaxprError, match="quiescent|pending|uninitialized|DMA"):
        normalize_view(adapt(source).view, budget=AnalysisBudget())


@pytest.mark.parametrize("mutation", ["alternatives", "elements", "bytes"])
def test_bounded_source_dma_evidence_rechecks_choices_and_transfer_count(mutation):
    source = paged_dma_source()
    result = compare_pallas_kernels(source, source, equivalence="semantic")
    proof = result.left_protocol
    assert proof is not None
    first, *rest = proof.projection.communication
    transfer = first.transfer
    assert transfer.source_alternatives
    fields = {
        "alternatives": {"source_alternatives": transfer.source_alternatives[:1]},
        "elements": {"elements": transfer.elements * 2},
        "bytes": {"logical_bytes": transfer.logical_bytes * 2},
    }[mutation]
    corrupt = replace(first, transfer=replace(transfer, **fields))
    forged = replace(
        result,
        left_protocol=replace(
            proof,
            projection=replace(proof.projection, communication=(corrupt, *rest)),
        ),
    )
    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(source, source, forged)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("separate_semaphores", [False, True])
def test_bounded_source_dmas_match_official_generic_cpu(separate_semaphores):
    function = paged_dma_function(separate_semaphores=separate_semaphores, interpret=True)
    for pages in ((0, 1), (1, 0), (1, 1)):
        inputs = paged_dma_inputs(pages)
        for output, reference in zip(function(*inputs), expected_pages(inputs), strict=True):
            np.testing.assert_array_equal(output, reference)


@pytest.mark.parametrize("separate_semaphores", [False, True])
def test_bounded_source_dmas_lower_to_verified_native_tpu_ir(separate_semaphores):
    with environment():
        lowered = (
            jax.jit(paged_dma_function(separate_semaphores=separate_semaphores))
            .trace(*PAGE_SIGNATURE)
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


def dynamic_function(mode="program", *, wrong=False, interpret=False):
    spec = pl.BlockSpec((8, 128), lambda p: (p, 0))
    scratch = (
        (tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.DMA)
        if mode.startswith("dma")
        else (tpu.VMEM((8, 128), jnp.float32),)
        if mode.startswith("path")
        else ()
    )

    def kernel(x, y, *resources):
        p = pl.program_id(0)
        raw = x[0, 0].astype(jnp.int32)
        guarded = jnp.minimum(jnp.maximum(raw, 0), 3)
        index = p if mode in ("program", "slice", "hole", "oob") else guarded
        if mode == "unguarded":
            index = raw
        if mode == "oob":
            index = p + 8
        if mode != "hole":
            y[...] = x[...]
        if mode.startswith("path"):
            resources[0][index, :] = x[0, :] + 10
            read_index = (index + 1) % 4 if mode == "path_hole" else index
            y[...] = jnp.broadcast_to(resources[0][read_index, :], (8, 128))
        elif mode.startswith("dma"):
            copy = tpu.async_copy(x, resources[0], resources[1])
            if mode == "dma_safe":
                copy.wait()
            resources[0][index, :] = x[0, :] + 10
            if mode == "dma_hazard":
                copy.wait()
            tpu.async_copy(resources[0], y, resources[1]).wait()
        elif mode.startswith("while"):
            count = guarded + (2 if wrong else 1)
            if mode == "while_select":
                count = jnp.where(raw > 0, 2 if wrong else 1, 3)

            def body(i):
                y[...] = y[...] + (i + 1).astype(jnp.float32)
                return i + 1

            lax.while_loop(lambda i: i < count, body, jnp.int32(0))
        elif mode == "slice":
            y[pl.ds(index, 2), :] = x[:2, :] + (11 if wrong else 10)
        else:
            y[index, :] = x[0, :] + (11 if wrong else 10)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
        grid=(2,),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
        scratch_shapes=scratch,
        interpret=interpret,
    )


def expected_dynamic(x, mode, *, wrong=False):
    result = x.copy()
    for p in range(2):
        block = result[p * 8 : (p + 1) * 8]
        raw = int(x[p * 8, 0])
        index = p if mode in ("program", "slice") else np.clip(raw, 0, 3)
        if mode == "path_init":
            block[...] = x[p * 8] + 10
        elif mode.startswith("while"):
            count = index + (2 if wrong else 1)
            if mode == "while_select":
                count = (2 if wrong else 1) if raw > 0 else 3
            for i in range(count):
                block += np.float32(i + 1)
        elif mode == "slice":
            block[index : index + 2] = x[p * 8 : p * 8 + 2] + (11 if wrong else 10)
        else:
            block[index] = x[p * 8] + (11 if wrong else 10)
    return result


def view_function(mode="read", *, wrong=False, interpret=False):
    shape = (16, 128)
    spec = pl.BlockSpec(shape, lambda: (0, 0))
    scratch = (
        (tpu.VMEM(shape, jnp.float32), tpu.SemaphoreType.DMA)
        if mode.startswith("dma")
        else (tpu.VMEM(shape, jnp.float32),)
        if mode == "hole"
        else ()
    )

    def kernel(x, y, *resources):
        if mode == "read":
            y[:8, :] = x.reshape((2, 8, 128))[1 if wrong else 0, :, :]
            y[8:, :] = x.reshape((2, 8, 128))[1, :, :]
        elif mode == "write":
            y.reshape((2, 8, 128))[0, :, :] = x[8:, :]
            y.reshape((2, 8, 128))[1, :, :] = x[8:, :] if wrong else x[:8, :]
        elif mode == "chain":
            y[:8, :] = x.reshape((2, 8, 128)).at[:, ::2, :].reshape((8, 128))[...]
            y[8:, :] = x.reshape((2, 8, 128)).at[:, 1::2, :].reshape((8, 128))[...]
        elif mode == "chain_write":
            y.reshape((2, 8, 128)).at[:, ::2, :].reshape((8, 128))[...] = x[:8, :]
            y.reshape((2, 8, 128)).at[:, 1::2, :].reshape((8, 128))[...] = x[8:, :]
        elif mode == "transpose_write":
            y[...] = x[...]
            y.reshape((2, 8, 128)).transpose((1, 0, 2)).at[2:6, :, :].reshape((8, 128))[...] = x[
                :8, :
            ]
        elif mode == "repeated":
            y.reshape((2, 8, 128)).at[jnp.zeros((2,), jnp.int32), :, :][...] = x[...].reshape(
                (2, 8, 128)
            )
        elif mode == "transpose":
            values = (
                x.reshape((2, 8, 128)).transpose((1, 0, 2)).at[2:6, :, :].reshape((8, 128))[...]
            )
            y[:8, :] = values
            y[8:, :] = values + 1
        elif mode == "hole":
            resources[0].reshape((2, 8, 128))[0, :, :] = x[:8, :]
            y[...] = resources[0][...]
        else:
            copy = tpu.async_copy(x.at[:8, :], resources[0].at[:8, :], resources[1])
            resources[0].reshape((2, 8, 128))[0 if mode == "dma_hazard" else 1, :, :] = x[8:, :]
            copy.wait()
            y[...] = resources[0][...]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=scratch,
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


def expected_view(x, mode):
    if mode in ("read", "dma_safe"):
        return x
    if mode == "write":
        return np.concatenate((x[8:], x[:8]))
    if mode == "chain":
        return np.concatenate((x[::2], x[1::2]))
    if mode == "chain_write":
        result = x.copy()
        result[::2], result[1::2] = x[:8], x[8:]
        return result
    if mode == "transpose_write":
        result = x.copy()
        result.reshape((2, 8, 128)).transpose((1, 0, 2))[2:6] = x[:8].reshape((4, 2, 128))
        return result
    values = x.reshape((2, 8, 128)).transpose((1, 0, 2))[2:6].reshape((8, 128))
    return np.concatenate((values, values + 1))


def project(function, *, limit=10000):
    source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((16, 128), jnp.float32))
    return normalize_view(adapt(source).view, budget=AnalysisBudget(limit)).projection


def layout_function(options=None):
    spec = pl.BlockSpec((16, 128), lambda: (0, 0))

    def kernel(x, y):
        y[...] = x[...] + 1

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(**(options or {})),
    )


@pytest.mark.parametrize(
    "mode", ["program", "slice", "bounded", "while_add", "while_select", "path_init", "dma_safe"]
)
@pytest.mark.parametrize("raw", [-5, 0, 2, 100])
def test_dynamic_source_bounds_preserve_values(mode, raw):
    projection = project(dynamic_function(mode))
    x = np.arange(16 * 128, dtype=np.float32).reshape((16, 128))
    x[0, 0], x[8, 0] = raw, -raw
    expected = expected_dynamic(x, mode)
    for row in range(16):
        for col in (0, 7, 127):
            assert at_point(projection.outputs[0], (x,), (row, col)) == expected[row, col]


@pytest.mark.parametrize("mode", ["program", "slice", "bounded", "while_add", "while_select"])
def test_dynamic_mutations_change_observable_outputs(mode):
    x = np.ones((16, 128), dtype=np.float32)
    left, right = [project(dynamic_function(mode, wrong=wrong)) for wrong in (False, True)]
    assert any(
        at_point(left.outputs[0], (x,), point) != at_point(right.outputs[0], (x,), point)
        for point in ((i, 0) for i in range(16))
    )


@pytest.mark.parametrize(
    "mode", ["read", "write", "chain", "chain_write", "transpose", "transpose_write", "dma_safe"]
)
def test_partial_views_keep_original_storage_values(mode):
    projection = project(view_function(mode))
    x = np.arange(16 * 128, dtype=np.float32).reshape((16, 128))
    expected = expected_view(x, mode)
    for point in np.ndindex(x.shape):
        assert at_point(projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("mode", ["hole", "dma_hazard"])
def test_partial_views_keep_holes_and_dma_conflicts(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized|DMA"):
        project(view_function(mode))


@pytest.mark.parametrize("mode", ["hole", "oob", "path_hole", "dma_hazard"])
def test_dynamic_write_does_not_invent_initialization_or_bounds(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized|bounds|quiescent"):
        project(dynamic_function(mode))


def test_partial_view_repeated_address_writes_are_not_admitted_as_bijections():
    with pytest.raises(UnsupportedJaxprError, match="scalar/slice"):
        project(view_function("repeated"))


def test_dynamic_axis_after_ref_reshape_preserves_selected_column():
    def kernel(x, y):
        column = jnp.minimum(jnp.maximum(x[0].astype(jnp.int32), 0), 3)
        y[...] = x.reshape((4, 128))[:, column]

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((4,), jnp.float32),
        grid=(),
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((512,), jnp.float32))
    projection = normalize_view(adapt(source).view, budget=AnalysisBudget()).projection
    x = np.arange(512, dtype=np.float32)
    x[0] = 2
    for row in range(4):
        assert at_point(projection.outputs[0], (x,), (row,)) == x.reshape((4, 128))[row, 2]


def test_tampered_view_base_index_cannot_use_view_coordinates_as_storage_coordinates():
    source = jax.make_jaxpr(view_function("read"))(jax.ShapeDtypeStruct((16, 128), jnp.float32))
    pending = [adapt(source).view.root]
    while pending:
        for operation in pending.pop().operations:
            if operation.call is not None:
                call = operation.call
                break
            pending.extend(binding.region for binding in operation.regions)
        else:
            continue
        break
    operations = list(call.kernel.operations)
    position = next(i for i, op in enumerate(operations) if op.access and op.access.transforms)
    access = operations[position].access
    operations[position] = replace(
        operations[position],
        access=replace(
            access,
            index=TensorIndex(
                (SliceIndex(0, 2, 1), SliceIndex(0, 8, 1), SliceIndex(0, 128, 1)), (2, 8, 128)
            ),
        ),
    )
    call = replace(call, kernel=replace(call.kernel, operations=tuple(operations)))
    with pytest.raises(UnsupportedJaxprError, match="original-storage full base index"):
        functionalize_tensor_call(call, budget=AnalysisBudget())


def test_dynamic_tail_dma_evidence_keeps_distinct_case_producers():
    projection = project(dynamic_function("dma_safe"))
    transfers = projection.communication
    assert len(transfers) == 5
    assert len({t.transfer.id for t in transfers}) == 5
    assert len({t.transfer.start.path for t in transfers}) == 5


def test_unconstrained_data_address_is_rejected_before_enumeration():
    with pytest.raises(UnsupportedJaxprError, match="address bounds"):
        project(dynamic_function("unguarded"))


def test_partial_view_footprint_is_budgeted_before_enumeration():
    with pytest.raises(ComparisonBudgetError, match="budget"):
        project(view_function("chain"), limit=100)


def test_large_but_in_bounds_dynamic_address_domain_is_charged_before_forking():
    def kernel(x, y):
        y[...] = x[...]
        index = jnp.minimum(jnp.maximum(x[0, 0].astype(jnp.int32), 0), 4095)
        y[index, :] = x[0, :]

    shape = (4096, 128)
    function = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32))
    source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct(shape, jnp.float32))
    with pytest.raises(ComparisonBudgetError, match="address_cases"):
        normalize_view(adapt(source).view, budget=AnalysisBudget(1000))


@pytest.mark.parametrize(
    "transforms",
    [
        (RefTransform("index", (), (SliceIndex(0, 16, 2), SliceIndex(0, 128, 1))),),
        (RefTransform("index", (), (16, SliceIndex(0, 128, 1))),),
        (RefTransform("reshape", (16, 128), (0,)),),
        (RefTransform("transpose", (0, 0)),),
        (RefTransform("reshape", (8, 128)),),
    ],
)
def test_tampered_ref_metadata_cannot_change_or_duplicate_storage(transforms):
    with pytest.raises(UnsupportedJaxprError):
        view_coordinates((16, 128), transforms, "tampered", AnalysisBudget())


def test_nondefault_layout_policy_is_admitted_and_retained_in_the_signature():
    normalized = []
    for enabled in (True, False):
        function = layout_function({"needs_layout_passes": enabled})
        source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((16, 128), jnp.float32))
        view = adapt(source).view
        normalized.append(normalize_view(view, budget=AnalysisBudget()))
    assert (
        normalized[0].boundaries[-1].compiler_options
        != normalized[1].boundaries[-1].compiler_options
    )
    assert dict((p.name, p.value) for p in normalized[1].boundaries[-1].compiler_options)[
        "needs_layout_passes"
    ] == (("type", "builtins", "bool"), False)
    x = np.arange(2048, dtype=np.float32).reshape((16, 128))
    for point in ((0, 0), (7, 127), (15, 127)):
        assert at_point(normalized[1].projection.outputs[0], (x,), point) == x[point] + 1


@pytest.mark.parametrize(
    "option", ["disable_semaphore_checks", "skip_device_barrier", "shape_invariant_numerics"]
)
def test_layout_review_does_not_admit_unreviewed_execution_flags(option):
    options = {"needs_layout_passes": False, option: option != "shape_invariant_numerics"}
    with pytest.raises(UnsupportedJaxprError, match="execution contract"):
        project(layout_function(options))


def test_reviewed_layout_option_lowers_to_verified_native_tpu_ir():
    with environment():
        lowered = (
            jax.jit(layout_function({"needs_layout_passes": False}))
            .trace(jax.ShapeDtypeStruct((16, 128), jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


@pytest.mark.parametrize(
    "mode", ["program", "slice", "bounded", "while_add", "while_select", "path_init", "dma_safe"]
)
def test_new_dynamic_kernels_lower_to_verified_native_tpu_ir(mode):
    with environment():
        lowered = (
            jax.jit(dynamic_function(mode))
            .trace(jax.ShapeDtypeStruct((16, 128), jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


@pytest.mark.parametrize("mode", ["read", "write", "dma_safe"])
def test_partial_reshape_kernels_lower_to_verified_native_tpu_ir(mode):
    with environment():
        lowered = (
            jax.jit(view_function(mode))
            .trace(jax.ShapeDtypeStruct((16, 128), jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


@pytest.mark.parametrize("mode", ["chain", "chain_write", "transpose", "transpose_write"])
def test_native_tpu_transform_limit_is_reported_separately(mode):
    with (
        environment(),
        pytest.raises(NotImplementedError, match="Strided slices|Unsupported transform"),
    ):
        jax.jit(view_function(mode)).trace(jax.ShapeDtypeStruct((16, 128), jnp.float32)).lower(
            lowering_platforms=("tpu",)
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "mode", ["program", "slice", "bounded", "while_add", "while_select", "path_init", "dma_safe"]
)
def test_dynamic_cases_match_official_generic_cpu(mode):
    x = np.arange(16 * 128, dtype=np.float32).reshape((16, 128))
    x[0, 0], x[8, 0] = -2, 2
    np.testing.assert_array_equal(
        dynamic_function(mode, interpret=True)(x), expected_dynamic(x, mode)
    )


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["read", "write", "chain", "chain_write", "dma_safe"])
def test_partial_views_match_official_generic_cpu(mode):
    x = np.arange(16 * 128, dtype=np.float32).reshape((16, 128))
    np.testing.assert_array_equal(view_function(mode, interpret=True)(x), expected_view(x, mode))
