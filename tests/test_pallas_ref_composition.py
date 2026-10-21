"""Ref chains retain address identity through bounded indices and local DMA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import ComparisonBudgetError, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import normalize_view
from tests.pallas_pipeline_cases import environment
from tests.pallas_projection_oracle import at_point


def transformed_function(mode="read", *, wrong=False, interpret=False):
    shape = (16, 128)
    spec = pl.BlockSpec(shape, lambda: (0, 0))
    scratch = (
        (tpu.VMEM(shape, jnp.float32), tpu.SemaphoreType.DMA)
        if mode.startswith("dma") or mode == "hole"
        else ()
    )

    def kernel(x, y, *resources):
        index = jnp.minimum(jnp.maximum(x[0, 0].astype(jnp.int32), 0), 1)
        if mode == "unbounded":
            index = x[0, 0].astype(jnp.int32)
        if mode == "oob":
            index += 2
        x_view = x.reshape((2, 8, 128))
        y_view = y.reshape((2, 8, 128))
        if mode in ("read", "oob", "unbounded"):
            y[...] = jnp.tile(x_view[index, :, :] + (1 if wrong else 0), (2, 1))
        elif mode == "write":
            y[...] = x[...]
            y_view[index, :, :] = x[8 if wrong else 0 : 16 if wrong else 8, :]
        elif mode == "slice":
            y[...] = x[...]
            y_view.at[pl.ds(index, 1), :, :].reshape((8, 128))[...] = x[:8, :] + (2 if wrong else 1)
        elif mode == "transpose":
            y[...] = jnp.tile(x_view.transpose((1, 0, 2))[:, index, :], (2, 1))
        elif mode == "hole":
            resources[0].reshape((2, 8, 128))[index, :, :] = x[:8, :]
            y[...] = resources[0][...]
        else:
            temp, sem = resources
            temp[...] = x[...]
            source = x_view.at[index, :, :]
            target = temp.reshape((2, 8, 128)).at[1 if wrong else 0, :, :]
            if mode == "dma_slice":
                source = x_view.at[pl.ds(index, 1), :, :].reshape((8, 1, 128))
                target = temp.at[:8, :].reshape((8, 1, 128))
            elif mode == "dma_transpose":
                source = x_view.transpose((1, 0, 2))
                target = temp.reshape((8, 2, 128))
            elif mode == "dma_scattered":
                source = x_view.at[:, ::2, :].reshape((8, 128))
            copy = tpu.async_copy(source, target, sem)
            if mode == "dma_alias":
                temp.reshape((2, 8, 128))[0, :, :] = x[:8, :]
            if mode == "dma_source_alias":
                # The target is a disjoint half, but the source alias is live.
                copy.wait()
                source = temp.reshape((2, 8, 128)).at[0, :, :]
                target = temp.reshape((2, 8, 128)).at[1, :, :]
                copy = tpu.async_copy(source, target, sem)
                temp[:8, :] = x[:8, :]
            if mode != "dma_no_wait":
                copy.wait()
            y[...] = temp[...]

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


def upstream_dma_reshape(*, slice_first=False, wrong=False, interpret=False):
    # jax_tpu_state_test.py::test_dma_reshaped_ref, retained in fixture snapshot.
    def kernel(x, y):
        def body(sem):
            source = (
                x.at[8 if wrong else 0 : 16 if wrong else 8, :, :].reshape((8, 128))
                if slice_first
                else x.reshape((16, 128)).at[8 if wrong else 0 : 16 if wrong else 8, :]
            )
            tpu.async_copy(source, y.reshape((8, 128)).at[...], sem).wait()

        pl.run_scoped(body, tpu.SemaphoreType.DMA)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 1, 128), jnp.float32),
        in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )


def project(function, shape=(16, 128), *, limit=30000):
    source = jax.make_jaxpr(function)(jax.ShapeDtypeStruct(shape, jnp.float32))
    return normalize_view(adapt(source).view, budget=AnalysisBudget(limit)).projection


def expected(x, mode, *, wrong=False):
    index = int(np.clip(x[0, 0], 0, 1))
    result = x.copy()
    if mode in ("read", "transpose"):
        return np.tile(x[index * 8 : (index + 1) * 8] + (1 if wrong else 0), (2, 1))
    if mode == "write":
        result[index * 8 : (index + 1) * 8] = x[8 if wrong else 0 : 16 if wrong else 8]
    elif mode == "slice":
        result[index * 8 : (index + 1) * 8] = x[:8] + (2 if wrong else 1)
    elif mode == "dma_transpose":
        result = x.reshape((2, 8, 128)).transpose((1, 0, 2)).reshape((16, 128))
    else:
        target = 1 if wrong else 0
        result[target * 8 : (target + 1) * 8] = x[index * 8 : (index + 1) * 8]
    return result


@pytest.mark.parametrize(
    "mode", ["read", "write", "slice", "transpose", "dma", "dma_slice", "dma_transpose"]
)
@pytest.mark.parametrize("raw", [-5, 1, 8])
def test_composed_views_preserve_dynamic_values(mode, raw):
    projection = project(transformed_function(mode))
    x = np.arange(2048, dtype=np.float32).reshape((16, 128))
    x[0, 0] = raw
    oracle = expected(x, mode)
    for row in range(16):
        for col in (0, 17, 127):
            assert at_point(projection.outputs[0], (x,), (row, col)) == oracle[row, col]


@pytest.mark.parametrize("mode", ["read", "write", "slice", "dma"])
def test_transformed_address_and_value_mutations_remain_observable(mode):
    x = np.arange(2048, dtype=np.float32).reshape((16, 128))
    x[0, 0] = 1
    left, right = [project(transformed_function(mode, wrong=w)) for w in (False, True)]
    assert any(
        at_point(left.outputs[0], (x,), (i, 17)) != at_point(right.outputs[0], (x,), (i, 17))
        for i in range(16)
    )


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("oob", "intermediate view"),
        ("unbounded", "address bounds"),
        ("hole", "uninitialized"),
        ("dma_alias", "DMA destination"),
        ("dma_source_alias", "DMA source"),
        ("dma_no_wait", "DMA destination"),
    ],
)
def test_transformed_views_do_not_hide_invalid_memory(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        project(transformed_function(mode))


def test_scattered_transformed_dma_preserves_each_selected_row():
    projection = project(transformed_function("dma_scattered"))
    x = np.arange(2048, dtype=np.float32).reshape(16, 128)
    expected = x.copy()
    expected[:8] = x.reshape(2, 8, 128)[:, ::2].reshape(8, 128)
    for point in np.ndindex(x.shape):
        assert at_point(projection.outputs[0], (x,), point) == expected[point]


def test_dynamic_ref_cases_are_budgeted():
    with pytest.raises(ComparisonBudgetError):
        project(transformed_function("write"), limit=20)


@pytest.mark.parametrize("slice_first", [False, True])
@pytest.mark.parametrize("wrong", [False, True])
def test_upstream_reshaped_dma_matches_independent_coordinates(slice_first, wrong):
    projection = project(upstream_dma_reshape(slice_first=slice_first, wrong=wrong), (16, 1, 128))
    x = np.arange(2048, dtype=np.float32).reshape((16, 1, 128))
    oracle = x[8:] if wrong else x[:8]
    for point in np.ndindex(oracle.shape):
        assert at_point(projection.outputs[0], (x,), point) == oracle[point]


@pytest.mark.parametrize("mode", ["read", "write", "slice", "dma", "dma_slice"])
def test_composed_sources_lower_to_verified_native_ir(mode):
    with environment():
        lowered = (
            jax.jit(transformed_function(mode))
            .trace(jax.ShapeDtypeStruct((16, 128), jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


@pytest.mark.parametrize("slice_first", [False, True])
def test_upstream_reshaped_dma_lowers_to_verified_native_ir(slice_first):
    with environment():
        lowered = (
            jax.jit(upstream_dma_reshape(slice_first=slice_first))
            .trace(jax.ShapeDtypeStruct((16, 1, 128), jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["read", "write", "slice", "dma", "dma_slice"])
def test_composed_sources_match_official_cpu(mode):
    x = np.arange(2048, dtype=np.float32).reshape((16, 128))
    x[0, 0] = 1
    np.testing.assert_array_equal(transformed_function(mode, interpret=True)(x), expected(x, mode))


@pytest.mark.pallas_communication
@pytest.mark.parametrize("slice_first", [False, True])
def test_upstream_reshaped_dma_matches_official_cpu(slice_first):
    x = np.arange(2048, dtype=np.float32).reshape((16, 1, 128))
    np.testing.assert_array_equal(
        upstream_dma_reshape(slice_first=slice_first, interpret=True)(x), x[:8]
    )


@pytest.mark.parametrize("dma", [False, True])
def test_dynamic_view_operands_survive_ssa_reordering_and_public_replay(dma):
    from jax_reorder import Status, compare_pallas_kernels, verify_pallas_kernel_comparison

    def build(reverse):
        def kernel(x, y, temp, sem):
            raw = x[0, 0].astype(jnp.int32)
            if reverse:
                delta = raw + 3
                index = jnp.minimum(jnp.maximum(raw, 0), 1)
            else:
                index = jnp.minimum(jnp.maximum(raw, 0), 1)
                delta = raw + 3
            if dma:
                tpu.async_copy(x.reshape((2, 8, 128)).at[index, :, :], temp.at[:8, :], sem).wait()
                result = temp[:8, :]
            else:
                result = x.reshape((2, 8, 128))[index, :, :]
            y[:8, :] = result + delta
            y[8:, :] = result

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
            scratch_shapes=(tpu.VMEM((16, 128), jnp.float32), tpu.SemaphoreType.DMA),
        )

    left, right = [
        jax.make_jaxpr(build(reverse))(jax.ShapeDtypeStruct((16, 128), jnp.float32))
        for reverse in (False, True)
    ]
    result = compare_pallas_kernels(left, right, check_protocol=True, analysis_budget=30000)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(left, right, result, analysis_budget=30000)


def test_transformed_dma_keeps_piecewise_source_values_in_logical_order():
    def kernel(x, y, temp, sem):
        temp[:8, :] = x[8:, :] + 1
        temp[8:, :] = x[:8, :] - 2
        tpu.async_copy(
            temp.reshape((2, 8, 128)).transpose((1, 0, 2)),
            y.reshape((8, 2, 128)),
            sem,
        ).wait()

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
        scratch_shapes=(tpu.VMEM((16, 128), jnp.float32), tpu.SemaphoreType.DMA),
    )
    projection = project(function)
    x = np.arange(2048, dtype=np.float32).reshape((16, 128))
    oracle = (
        np.concatenate((x[8:] + 1, x[:8] - 2))
        .reshape((2, 8, 128))
        .transpose((1, 0, 2))
        .reshape((16, 128))
    )
    for point in np.ndindex(x.shape):
        assert at_point(projection.outputs[0], (x,), point) == oracle[point]


@pytest.mark.parametrize("mode", ["transpose", "dma_transpose"])
def test_native_transpose_ref_rejection_is_separate_from_semantic_projection(mode):
    with environment(), pytest.raises(NotImplementedError, match="Unsupported transform"):
        jax.jit(transformed_function(mode)).trace(
            jax.ShapeDtypeStruct((16, 128), jnp.float32)
        ).lower(lowering_platforms=("tpu",))
