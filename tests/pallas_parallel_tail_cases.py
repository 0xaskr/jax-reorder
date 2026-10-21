"""Real Pallas sources for parallel full/tail owners with serial ANY revisits."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

ROWS = 8
WIDTH = 256


def function(
    mode="partial",
    *,
    columns=384,
    multiple=False,
    different_tail=False,
    reverse=False,
    interpret=False,
):
    groups = (columns + WIDTH - 1) // WIDTH

    def target(p):
        return groups - 1 - p if reverse else p

    def kernel(x, y, *others):
        p, t = pl.program_id(0), pl.program_id(1)

        @pl.when(t < 2)
        def initialize():
            if mode == "missing_init":
                y[:, :128] = x[:, :128]
            else:
                y[...] = x[...]

        @pl.when(t == 2)
        def revisit():
            if mode == "padding_leak":
                y[:, :128] = y[:, 128:] + x[:, :128]
            elif mode == "reset":
                y[:, :128] = x[:, :128]
            elif mode == "full":

                @pl.when((target(p) + 1) * WIDTH <= columns)
                def complete_owner():
                    y[...] += x[...]

                if columns % WIDTH:

                    @pl.when((target(p) + 1) * WIDTH > columns)
                    def tail_owner():
                        y[:, : columns % WIDTH] += x[:, : columns % WIDTH]
            else:
                y[:, :128] += x[:, :128]

        if multiple:
            total = others[0]

            @pl.when(t == 0)
            def initialize_total():
                total[...] = jnp.zeros((ROWS, WIDTH), jnp.float32)

            total[...] += x[...]

    result = jax.ShapeDtypeStruct((2 * ROWS, columns), jnp.float32)

    def output_mapping(p, t):
        owner = jnp.where(t == 2, groups - 1 - p, p) if mode == "owner_swap" else target(p)
        return t % 2, 0 if mode == "overlap" else owner

    output = pl.BlockSpec(
        (ROWS, WIDTH),
        output_mapping,
        pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
    )
    return pl.pallas_call(
        kernel,
        out_shape=(
            result,
            jax.ShapeDtypeStruct(
                (ROWS, columns - 64 if different_tail else groups * WIDTH), jnp.float32
            ),
        )
        if multiple
        else result,
        grid=(groups, 3),
        in_specs=[pl.BlockSpec((ROWS, WIDTH), lambda p, t: (t, p))],
        out_specs=(
            output,
            pl.BlockSpec(
                (ROWS, WIDTH),
                lambda p, t: (0, groups - 1 - p if different_tail else p),
                pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY)
                if different_tail
                else None,
            ),
        )
        if multiple
        else output,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
        interpret=interpret,
    )


def expected(
    x, mode="partial", *, columns=384, multiple=False, different_tail=False, reverse=False
):
    x = np.asarray(x)
    output = np.empty((2 * ROWS, columns), x.dtype)
    groups = x.shape[1] // WIDTH
    for p in range(groups):
        target = groups - 1 - p if reverse else p
        start, stop = target * WIDTH, min((target + 1) * WIDTH, columns)
        count = stop - start
        output[:, start:stop] = x[: 2 * ROWS, p * WIDTH : p * WIDTH + count]
        changed = count if mode == "full" else min(count, 128)
        extra = x[2 * ROWS :, p * WIDTH : p * WIDTH + changed]
        if mode == "reset":
            output[:ROWS, start : start + changed] = extra
        else:
            output[:ROWS, start : start + changed] += extra
    values = (output,)
    if multiple:
        total = x.reshape(3, ROWS, x.shape[1]).sum(axis=0)
        if different_tail:
            total = total.reshape(ROWS, groups, WIDTH)[:, ::-1].reshape(ROWS, -1)[:, : columns - 64]
        values += (total,)
    return values


def two_tail_function(*, mixed=False, reverse=False, interpret=False):
    """Two tail dimensions share a flattened or separate parallel owner."""

    def coordinates(*point):
        if mixed:
            p, t = point
            row, col = p // 2, p % 2
        else:
            row, col, t = point
        return (1 - row, 1 - col, t) if reverse else (row, col, t)

    def kernel(x, y):
        t = pl.program_id(1 if mixed else 2)

        @pl.when(t < 2)
        def initialize():
            y[...] = x[...]

        @pl.when(t == 2)
        def revisit():
            y[:, :4, :128] += x[:, :4, :128]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 12, 384), jnp.float32),
        grid=(4, 3) if mixed else (2, 2, 3),
        in_specs=[
            pl.BlockSpec((1, ROWS, WIDTH), lambda *p: (coordinates(*p)[2], *coordinates(*p)[:2]))
        ],
        out_specs=pl.BlockSpec(
            (1, ROWS, WIDTH),
            lambda *p: (coordinates(*p)[2] % 2, *coordinates(*p)[:2]),
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
        ),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
            if mixed
            else ("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )


def two_tail_expected(x):
    output = np.asarray(x)[:2, :12, :384].copy()
    for row in (0, 8):
        for col in (0, 256):
            output[0, row : row + 4, col : col + 128] += x[2, row : row + 4, col : col + 128]
    return (output,)


def dma_function(*, missing_wait=False):
    def kernel(x, y, sem):
        t = pl.program_id(1)

        @pl.when(t < 2)
        def initialize():
            y[...] = x[...]

        @pl.when(t == 2)
        def revisit():
            transfer = tpu.async_copy(x.at[:, :128], y.at[:, :128], sem)
            if not missing_wait:
                transfer.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 384), jnp.float32),
        grid=(2, 3),
        in_specs=[pl.BlockSpec((ROWS, WIDTH), lambda p, t: (t, p))],
        out_specs=pl.BlockSpec(
            (ROWS, WIDTH),
            lambda p, t: (t % 2, p),
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
        ),
        scratch_shapes=[tpu.SemaphoreType.DMA],
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )
