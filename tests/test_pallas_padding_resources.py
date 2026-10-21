"""Explicitly initialized padded values can participate in scratch and DMA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import UnsupportedJaxprError, compare_pallas_kernels
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import normalize_view
from tests.pallas_projection_oracle import at_point


def padded_resources(*, mode="normal", rank=2, reduce=False):
    shape = (8, 233) if rank == 2 else (233,)
    block = (8, 128) if rank == 2 else (128,)
    output_shape = block if reduce else shape
    spec = pl.BlockSpec(block, lambda p: (0, p) if rank == 2 else (p,))
    output_spec = pl.BlockSpec(block, lambda p: (0,) * rank) if reduce else spec

    def kernel(x, y, a, b, sem):
        p = pl.program_id(0)
        mask = jnp.broadcast_to(p * 128 + jnp.arange(128) < (232 if mode == "gap" else 233), block)
        if mode == "input_dma":
            tpu.async_copy(x, a, sem).wait()
        elif mode == "unmasked":
            a[...] = x[...]
        elif mode != "uninitialized":
            a[...] = pp.load(x, (slice(None),) * rank, mask=mask, other=7)
        transfer = tpu.async_copy(a, b, sem)
        if mode == "source_write":
            a[...] = jnp.zeros(block, jnp.float32)
        if mode != "missing_wait":
            transfer.wait()
        if reduce:

            @pl.when(p == 0)
            def initialize():
                y[...] = jnp.zeros(block, jnp.float32)

            y[...] += jnp.broadcast_to(jnp.sum(b[...], axis=-1, keepdims=True), block)
        elif mode == "direct_output":
            tpu.async_copy(b, y, sem).wait()
        else:
            pp.store(y, (slice(None),) * rank, b[...] + 1, mask=mask)

    return (
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(output_shape, jnp.float32),
            grid=(2,),
            in_specs=[spec],
            out_specs=output_spec,
            scratch_shapes=[
                tpu.VMEM(block, jnp.float32),
                tpu.VMEM(block, jnp.float32),
                tpu.SemaphoreType.DMA,
            ],
            compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        ),
        shape,
        output_shape,
    )


def proof(**options):
    fn, shape, output_shape = padded_resources(**options)
    closed = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))
    return (
        normalize_view(adapt(closed).view, budget=AnalysisBudget(100000)),
        shape,
        output_shape,
        closed,
    )


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize(
    "mode,reduce", [("normal", False), ("direct_output", False), ("normal", True)]
)
def test_initialized_mask_tail_can_flow_through_scratch_and_dma(rank, mode, reduce):
    evidence, shape, output_shape, closed = proof(rank=rank, mode=mode, reduce=reduce)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    padded = np.pad(x, [(0, 0)] * (rank - 1) + [(0, 23)], constant_values=-10000)
    expected = (
        np.broadcast_to(x.sum(axis=-1, keepdims=True) + 23 * 7, output_shape)
        if reduce
        else x + (mode != "direct_output")
    )
    for point in np.ndindex(output_shape):
        assert at_point(evidence.projection.outputs[0], (padded,), point) == expected[point]
    assert len(evidence.projection.communication) == (4 if mode == "direct_output" else 2)
    compare_pallas_kernels(closed, closed, check_protocol=True, analysis_budget=100000)


@pytest.mark.parametrize(
    "mode", ["input_dma", "unmasked", "gap", "uninitialized", "source_write", "missing_wait"]
)
def test_padding_does_not_hide_missing_initialization_or_dma_hazards(mode):
    with pytest.raises(UnsupportedJaxprError):
        proof(mode=mode)


def aligned_tail(*, mode="normal", reduce=False):
    """Native TPU form: initialize padding explicitly, then DMA valid rectangles."""
    shape, block = (8, 384), (8, 256)
    output_shape = (8, 128) if reduce else shape
    spec = pl.BlockSpec(block, lambda p: (0, p))
    output_spec = pl.BlockSpec((8, 128), lambda p: (0, 0)) if reduce else spec

    def kernel(x, y, a, b, sem):
        p = pl.program_id(0)
        if mode == "stale_tail":

            @pl.when(p == 0)
            def initialize_only_first():
                a[...] = jnp.full(block, 7, jnp.float32)
        else:
            a[...] = jnp.full(block, 7, jnp.float32)

        def load(width):
            transfer = tpu.async_copy(x.at[:, :width], a.at[:, :width], sem)
            if mode != "missing_wait":
                transfer.wait()

        if mode == "unguarded":
            load(256)
        else:
            jax.lax.cond(p == 0, lambda: load(256), lambda: load(128))
        tpu.async_copy(a, b, sem).wait()
        if reduce:

            @pl.when(p == 0)
            def initialize_output():
                y[...] = jnp.zeros((8, 128), jnp.float32)

            y[...] += jnp.broadcast_to(jnp.sum(b[...], axis=-1, keepdims=True), (8, 128))
        else:
            tpu.async_copy(b, y, sem).wait()

    return (
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(output_shape, jnp.float32),
            grid=(2,),
            in_specs=[spec],
            out_specs=output_spec,
            scratch_shapes=[
                tpu.VMEM(block, jnp.float32),
                tpu.VMEM(block, jnp.float32),
                tpu.SemaphoreType.DMA,
            ],
            compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        ),
        shape,
        output_shape,
    )


@pytest.mark.parametrize("reduce", [False, True])
def test_native_aligned_tail_keeps_explicit_padding_values_in_reduction(reduce):
    fn, shape, output_shape = aligned_tail(reduce=reduce)
    source = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))
    evidence = normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    # The independent oracle evaluates both select arms eagerly, so sentinel
    # storage is supplied solely for its unused branch; padding is never read.
    padded = np.pad(x, ((0, 0), (0, 128)), constant_values=-10000)
    expected = (
        np.broadcast_to(x.sum(axis=-1, keepdims=True) + 128 * 7, output_shape) if reduce else x
    )
    for row in (0, 7):
        for col in range(output_shape[1]):
            assert (
                at_point(evidence.projection.outputs[0], (padded,), (row, col))
                == expected[row, col]
            )
    compare_pallas_kernels(source, source, check_protocol=True, analysis_budget=100000)


@pytest.mark.parametrize(
    "mode,reason", [("unguarded", "undefined input padding"), ("missing_wait", "cumulative DMA")]
)
def test_native_tail_dma_guards_and_waits_cannot_be_omitted(mode, reason):
    fn, shape, _ = aligned_tail(mode=mode)
    source = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(source).view, budget=AnalysisBudget(100000))


def test_scratch_tail_retains_its_old_value_when_not_reinitialized():
    fn, shape, _ = aligned_tail(mode="stale_tail", reduce=True)
    source = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))
    evidence = normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = x.sum(axis=-1) + x[:, 128:256].sum(axis=-1)
    assert (
        at_point(
            evidence.projection.outputs[0],
            (np.pad(x, ((0, 0), (0, 128)), constant_values=-10000),),
            (0, 0),
        )
        == expected[0]
    )
    assert expected[0] != x[0].sum() + 128 * 7
