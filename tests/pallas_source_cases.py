"""Native Pallas source cases shared by semantic tests and offline compilation."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def squeezed_call(shape, squeezed, *, retain_axes=False, wrong=False, interpret=False):
    grid = tuple(shape[i] for i in squeezed)
    block = tuple((1 if retain_axes else None) if i in squeezed else n for i, n in enumerate(shape))

    def mapping(*programs):
        return tuple(programs[squeezed.index(i)] if i in squeezed else 0 for i in range(len(shape)))

    def kernel(x, y):
        y[...] = x[...] + (2 if wrong else 1)

    spec = pl.BlockSpec(block, mapping)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        interpret=interpret,
    )


def value_call(operation, *, reference=False, wrong=False, interpret=False):
    output_shape = (8, 128) if operation in ("stop_gradient", "slice", "slice_stride") else (128,)
    dtype = jnp.bool_ if operation in ("reduce_and", "reduce_or") else jnp.float32

    def kernel(x, y):
        value = x[...]
        if operation == "stop_gradient":
            result = value[:, :128] * 2
            if not reference:
                result = lax.stop_gradient(result)
        elif operation == "slice":
            start = 0 if wrong else 128
            result = x[:, start : start + 128] if reference else value[:, 128:256]
        elif operation == "slice_stride":
            result = x[:, 1 if wrong else 0 : 256 : 2] if reference else value[:, ::2]
        elif operation == "reduce_min":
            value = value[:, :128]
            result = -jnp.max(-value, axis=0) if reference else jnp.min(value, axis=0)
        elif operation == "reduce_and":
            value = value[:, :128] > 0
            result = ~jnp.any(~value, axis=0) if reference else jnp.all(value, axis=0)
        elif operation == "reduce_or":
            value = value[:, :128] > 0
            result = ~jnp.all(~value, axis=0) if reference else jnp.any(value, axis=0)
        else:
            raise ValueError(operation)
        if wrong and operation not in ("slice", "slice_stride"):
            result = ~result if dtype == jnp.bool_ else result + 1
        y[...] = result

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        interpret=interpret,
    )


def prefetch_call(
    mode="counter",
    *,
    parallel=False,
    indexed_prefetch=False,
    indexed_reference=False,
    indexed_output=False,
    indexed_output_reference=False,
    initial=0,
    step=1,
    programs=3,
    output_blocks=3,
    output_wrap=False,
    output_revisit="any",
    accumulate_output=False,
    multiple_outputs=False,
    dynamic_initial=False,
    interpret=False,
):
    def kernel(counter, x, y, *others):
        if mode == "formula":
            count = initial + (pl.program_id(0) + 1) * step
        elif mode == "reference":
            count = step + counter[0]
            counter[0] = count
        else:
            if mode == "reset_each":
                counter[0] = initial
            counter[0] += step
            count = counter[0]
        value = x[...] + count
        if accumulate_output:

            @pl.when(pl.program_id(0) < output_blocks)
            def initialize():
                y[...] = jnp.zeros_like(y)

            y[...] += value
        else:
            y[...] = value
        if multiple_outputs:
            total = others[0]

            @pl.when(pl.program_id(0) == 0)
            def initialize_total():
                total[...] = jnp.zeros_like(total)

            total[...] += value

    spec = pl.BlockSpec((128,), lambda i, _: (i,))
    state_spec = pl.BlockSpec((128,), lambda _, counter: (counter[0],))
    if indexed_prefetch:
        input_spec = state_spec
    elif indexed_reference:
        input_spec = pl.BlockSpec((128,), lambda i, _: (initial + i * step,))
    else:
        input_spec = spec

    def output_map(i, counter):
        index = counter[0] if indexed_output else initial + i * step
        return (index % output_blocks if output_wrap else index,)

    output_spec = (
        pl.BlockSpec(
            (128,),
            output_map,
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode(output_revisit))
            if output_wrap and output_revisit
            else None,
        )
        if indexed_output or indexed_output_reference
        else spec
    )
    output_shape = jax.ShapeDtypeStruct((128 * output_blocks,), jnp.float32)
    call = pl.pallas_call(
        kernel,
        out_shape=(output_shape, jax.ShapeDtypeStruct((128,), jnp.float32))
        if multiple_outputs
        else output_shape,
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(programs,),
            in_specs=[input_spec],
            out_specs=(output_spec, pl.BlockSpec((128,), lambda i, _: (0,)))
            if multiple_outputs
            else output_spec,
        ),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel" if parallel else "arbitrary",)
        ),
        interpret=interpret,
    )
    return call if dynamic_initial else lambda x: call(jnp.full((1,), initial, jnp.int32), x)


def stateful_jit_call(mode="jit", *, interpret=False):
    def kernel(x, y):
        def write():
            y[...] = x[...]
            y[...] = x[...] + (2 if mode == "wrong" else 1)

        if mode == "jit":
            jax.jit(write)()
        else:
            write()

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((128,), jnp.float32), interpret=interpret
    )


def weak_cond_call(*, reference=False, interpret=False):
    def kernel(x, y):
        condition = x[0] > 0
        selected = (
            lax.select(condition, jnp.int32(1), jnp.int32(0))
            if reference
            else lax.cond(condition, lambda: jnp.int32(1), lambda: 0)
        )
        y[...] = x[...] + selected

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((128,), jnp.float32), interpret=interpret
    )


def concatenate_call(axis, mode="concat", *, interpret=False):
    shape = (16, 128) if axis == 0 else (8, 256)

    def kernel(a, b, y):
        if mode == "concat":
            y[...] = jnp.concatenate((a[...], b[...]), axis=axis)
        else:
            first, second = (b, a) if mode == "swap" else (a, b)
            if axis == 0:
                y[:8, :] = first[...]
                y[8:, :] = second[...]
            else:
                y[:, :128] = first[...]
                y[:, 128:] = second[...]

    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32), interpret=interpret
    )
