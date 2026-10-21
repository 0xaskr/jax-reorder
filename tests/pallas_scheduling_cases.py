"""Native scheduling source cases shared by semantic tests and offline AOT."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def interleaved(*, reorder=False, delta=0, persistent=False, dma=False, interpret=False, owners=2):
    def body(x, y, scratch, *sems):
        serial = pl.program_id(1 if reorder else 0)
        if persistent:

            @pl.when(serial == 0)
            def initialize():
                scratch[...] = x[...]
        elif dma:
            tpu.make_async_copy(x, scratch, sems[0]).start()
            tpu.make_async_copy(x, scratch, sems[0]).wait()
        else:
            scratch[...] = x[...]
        scratch[...] += serial + delta
        y[...] = scratch[...]

    index_map = (lambda p, a: (p, a)) if reorder else (lambda a, p: (p, a))
    spec = pl.BlockSpec((8, 128), index_map)
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8 * owners, 384), jnp.float32),
        grid=(owners, 3) if reorder else (3, owners),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32)] + ([tpu.SemaphoreType.DMA] if dma else []),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary") if reorder else ("arbitrary", "parallel")
        ),
        interpret=interpret,
    )


def routed(*, mode="forward", target=1, size=2, serial=False, scratch=False):
    def body(x, y, sem, *scratch_refs):
        p = pl.program_id(0)
        if mode == "cycle":
            pl.semaphore_wait(sem, 1)
            pl.semaphore_signal(sem, 1, core_index=1 - p)
        elif mode == "self":
            pl.semaphore_signal(sem, 1, core_index=p)
            pl.semaphore_wait(sem, 1)
        elif mode == "read":
            value = pl.semaphore_read(sem)
            y[...] = x[...] + value
            return
        elif mode == "data_branch":

            @pl.when(x[0, 0] > 0)
            def signal():
                pl.semaphore_signal(sem, 1, core_index=target)

            pl.semaphore_wait(sem, 1)
        else:

            @pl.when(p == 0)
            def signal():
                pl.semaphore_signal(sem, 2 if mode == "wrong_count" else 1, core_index=target)

            @pl.when(p == 1)
            def wait():
                pl.semaphore_wait(sem, 1)

        if scratch:

            @pl.when(p == 0)
            def initialize():
                scratch_refs[0][...] = x[...]

            y[...] = scratch_refs[0][...]
        else:
            y[...] = x[...] * 2

    spec = pl.BlockSpec((8, 128), lambda p: (0, p))
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8, 128 * size), jnp.float32),
        grid=(size,),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[tpu.SemaphoreType.REGULAR]
        + ([tpu.VMEM((8, 128), jnp.float32)] if scratch else []),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("arbitrary" if serial else "parallel",)
        ),
    )


def banked_grid(*, reorder=False, mode="base", owners=2):
    def body(x, y, scratch, *sems):
        a = pl.program_id(1 if reorder else 0)
        p = pl.program_id(0 if reorder else 1)
        bank = 0 if mode == "overlap" else p
        ref = scratch if reorder else scratch.at[bank]

        @pl.when((a == 0) & ((p == 0) if mode == "missing_init" else True))
        def initialize():
            if mode in ("dma", "pending"):
                tpu.make_async_copy(x, ref, sems[0]).start()
                if mode != "pending":
                    tpu.make_async_copy(x, ref, sems[0]).wait()
            else:
                ref[...] = x[...]

        ref[...] += a + 1
        y[...] = ref[...]

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8 * owners, 384), jnp.float32),
        grid=(owners, 3) if reorder else (3, owners),
        in_specs=[
            pl.BlockSpec((8, 128), (lambda p, a: (p, 0)) if reorder else (lambda a, p: (p, 0)))
        ],
        out_specs=pl.BlockSpec(
            (8, 128), (lambda p, a: (p, a)) if reorder else (lambda a, p: (p, a))
        ),
        scratch_shapes=[tpu.VMEM((8, 128) if reorder else (owners, 8, 128), jnp.float32)]
        + ([tpu.SemaphoreType.DMA] if mode in ("dma", "pending") else []),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary") if reorder else ("arbitrary", "parallel")
        ),
    )


def contiguous_suffix(*, reorder=False):
    def body(x, y, scratch):
        k = pl.program_id(2)

        @pl.when(k == 0)
        def initialize():
            scratch[...] = jnp.zeros((8, 128), jnp.float32)

        scratch[...] += x[...]
        y[...] = scratch[...]

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2, 3),
        in_specs=[
            pl.BlockSpec(
                (8, 128),
                (lambda p, a, k: (p, a * 3 + k)) if reorder else (lambda a, p, k: (p, a * 3 + k)),
            )
        ],
        out_specs=pl.BlockSpec(
            (8, 128), (lambda p, a, k: (p, a)) if reorder else (lambda a, p, k: (p, a))
        ),
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32)],
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
            if reorder
            else ("arbitrary", "parallel", "arbitrary")
        ),
    )
