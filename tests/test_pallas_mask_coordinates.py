"""Masks over squeezed access axes must be proved in the original Ref coordinates."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import UnsupportedJaxprError, compare_pallas_kernels
from tests.pallas_projection_oracle import at_point


def function(mode="tail", *, middle=False, interpret=False):
    width = 100 if mode in ("tail", "reduce") else 128
    shape = (8, 1, width) if middle else (1, width)
    block = (8, 1, 128) if middle else (1, 128)
    result_shape = (8, 128) if middle else (128,)
    output_shape = (1, 1) if mode == "reduce" else shape
    spec = pl.BlockSpec(block, lambda _: (0,) * len(block))
    output = pl.BlockSpec((1, 1), lambda _: (0, 0)) if mode == "reduce" else spec

    def kernel(x, y):
        zero = pl.program_id(0)
        index = (slice(None), zero, slice(None)) if middle else (zero, slice(None))
        limit = 64 if mode in ("gap_store", "gap_load", "uninitialized_store") else width
        mask = jnp.broadcast_to(jnp.arange(128) < limit, result_shape)
        if mode in ("gap_store", "uninitialized_store"):
            if mode == "gap_store":
                y[...] = x[...]
            pp.store(y, index, x[index] + 100, mask=mask)
            return
        values = pp.load(x, index, mask=mask, other=7)
        if mode == "reduce":
            y[zero, zero] = jnp.sum(values)
        else:
            pp.store(
                y,
                index,
                values + 1,
                mask=jnp.ones(result_shape, dtype=jnp.bool_) if mode == "gap_load" else mask,
            )

    return (
        pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(output_shape, jnp.float32),
            grid=(1,),
            in_specs=(spec,),
            out_specs=output,
            compiler_params=tpu.CompilerParams(),
            interpret=interpret,
        ),
        shape,
        output_shape,
    )


def projection(mode, *, middle=False):
    f, shape, output_shape = function(mode, middle=middle)
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct(shape, jnp.float32))
    proof = compare_pallas_kernels(source, source, check_protocol=True).left_protocol.projection
    return proof.outputs[0], shape, output_shape


@pytest.mark.parametrize("middle", [False, True])
def test_squeezed_axis_tail_masks_preserve_valid_read_and_write_coordinates(middle):
    value, shape, _ = projection("tail", middle=middle)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    for point in np.ndindex(shape):
        assert at_point(value, (x,), point) == x[point] + 1


def test_squeezed_load_keeps_result_coordinates_for_its_other_value():
    value, shape, _ = projection("reduce")
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    # The NumPy oracle evaluates select operands eagerly. Sentinel padding
    # makes the unselected input lanes readable and detects a misplaced mask.
    padded = np.pad(x, ((0, 0), (0, 28)), constant_values=-1000)
    assert at_point(value, (padded,), (0, 0)) == x.sum() + 28 * 7


@pytest.mark.parametrize("middle", [False, True])
@pytest.mark.parametrize("mode", ["gap_store", "gap_load"])
def test_squeezed_axis_partial_masks_preserve_inactive_values(mode, middle):
    value, shape, _ = projection(mode, middle=middle)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = (
        np.where(np.arange(128) < 64, x + 100, x)
        if mode == "gap_store"
        else np.where(np.arange(128) < 64, x + 1, 8)
    )
    for point in np.ndindex(shape):
        assert at_point(value, (x,), point) == expected[point]


@pytest.mark.parametrize("middle", [False, True])
def test_squeezed_axis_partial_store_cannot_initialize_masked_off_lanes(middle):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized|undefined"):
        projection("uninitialized_store", middle=middle)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["tail", "reduce", "gap_store", "gap_load"])
def test_mask_coordinate_regressions_have_official_cpu_oracles(mode):
    f, shape, _ = function(mode, interpret=True)
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    actual = np.asarray(f(x))
    if mode == "reduce":
        expected = np.asarray([[x.sum() + 28 * 7]], dtype=np.float32)
    elif mode == "gap_store":
        expected = np.where(np.arange(128)[None, :] < 64, x + 100, x)
    elif mode == "gap_load":
        expected = np.where(np.arange(128)[None, :] < 64, x + 1, 8)
    else:
        expected = x + 1
    np.testing.assert_array_equal(actual, expected)
