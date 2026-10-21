"""Cross-path sources for restored output windows and matrix remote DMA."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

ROWS = 8
WIDTH = 256


def local_padded_revisit(*, wrong=False, missing_init=False, interpret=False):
    def kernel(x, y, total):
        p = pl.program_id(0)

        @pl.when(p == 0)
        def initialize():
            if missing_init:
                y[:, :128] = x[:, :128]
            else:
                y[...] = x[...]
            total[...] = jnp.zeros((ROWS, 128), jnp.float32)

        @pl.when(p == 1)
        def tail_window():
            y[:, :128] = x[:, :128]

        @pl.when(p == 2)
        def revisit():
            y[:, :128] += x[:, :128] * (2 if wrong else 1)

        total[...] += x[:, :128]

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((ROWS, 384), jnp.float32),
            jax.ShapeDtypeStruct((ROWS, 128), jnp.float32),
        ),
        grid=(3,),
        in_specs=(pl.BlockSpec((ROWS, WIDTH), lambda p: (0, p)),),
        out_specs=(
            pl.BlockSpec(
                (ROWS, WIDTH),
                lambda p: (0, p % 2),
                pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
            ),
            pl.BlockSpec((ROWS, 128), lambda p: (0, 0)),
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )


def local_padded_expected(x, *, wrong=False):
    x = np.asarray(x)
    result = np.concatenate((x[:, :256], x[:, 256:384]), axis=1).copy()
    result[:, :128] += x[:, 512:640] * (2 if wrong else 1)
    total = sum(x[:, p * WIDTH : p * WIDTH + 128] for p in range(3))
    return result, total


def matrix_revisit(
    *,
    remote=True,
    self_target=False,
    wrong=False,
    synchronize=True,
    receive_first=False,
    mesh=None,
    sharded=True,
):
    def kernel(x, y, *semaphores):
        p = pl.program_id(0)
        if remote:
            send, receive, ack = semaphores
            device = jax.lax.axis_index("d")
            peer = device if self_target else 1 - device
            if synchronize:

                @pl.when(p > 0)
                def acknowledge():
                    pl.semaphore_signal(ack, 1, device_id=(peer,))
                    pl.semaphore_wait(ack, 1)

            def copy(source, destination):
                dma = tpu.async_remote_copy(source, destination, send, receive, (peer,))
                if receive_first:
                    dma.wait_recv()
                    dma.wait_send()
                else:
                    dma.wait()

            copy(x.at[:, :128] if wrong else x.at[:, 128:], y.at[:, :128])

            @pl.when(p < 2)
            def first_visits():
                copy(x.at[:, :128], y.at[:, 128:])

        else:
            y[:, :128] = x[:, :128] if wrong else x[:, 128:]

            @pl.when(p < 2)
            def first_visits():
                y[:, 128:] = x[:, :128]

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((ROWS, 2 * WIDTH), jnp.float32),
        grid=(3,),
        in_specs=(pl.BlockSpec((ROWS, WIDTH), lambda p: (0, p)),),
        out_specs=pl.BlockSpec(
            (ROWS, WIDTH),
            lambda p: (0, p % 2),
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
        ),
        scratch_shapes=(
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR,
        )
        if remote
        else (),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    if not sharded:
        return call
    if mesh is None:
        mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    return jax.shard_map(
        call,
        mesh=mesh,
        in_specs=P(None, "d"),
        out_specs=P(None, "d"),
        check_vma=False,
    )


def matrix_expected(x, *, remote=True, self_target=False, wrong=False, **unused):
    source = np.stack(np.split(np.asarray(x), 2, axis=1))
    if remote and not self_target:
        source = source[::-1]
    result = np.empty((2, ROWS, 2 * WIDTH), np.float32)
    for p in range(3):
        base = (p % 2) * WIDTH
        source_base = p * WIDTH
        result[:, :, base : base + 128] = source[
            :, :, source_base + (0 if wrong else 128) : source_base + (128 if wrong else WIDTH)
        ]
        if p < 2:
            result[:, :, base + 128 : base + WIDTH] = source[:, :, source_base : source_base + 128]
    return np.concatenate(tuple(result), axis=1)


def padded_temporary_lifetime(*, consecutive=False, interpret=False):
    def kernel(x, y):
        p = pl.program_id(0)

        @pl.when(p == 0)
        def initialize_padded_window():
            y[...] = jnp.full((ROWS, WIDTH), 7, jnp.float32)

        @pl.when(p == (1 if consecutive else 2))
        def read_temporary_padding():
            y[:, :128] = y[:, 128:] + 1

        @pl.when(p == (2 if consecutive else 1))
        def other_window():
            y[...] = x[...]

    def mapping(p):
        return (0, jnp.where(p < 2, 1, 0) if consecutive else 1 - p % 2)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((ROWS, 384), jnp.float32),
        grid=(3,),
        in_specs=(pl.BlockSpec((ROWS, WIDTH), lambda p: (0, p)),),
        out_specs=pl.BlockSpec(
            (ROWS, WIDTH),
            mapping,
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )
