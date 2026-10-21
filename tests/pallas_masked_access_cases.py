"""Ordinary Pallas masked accesses, staged tails and local DMA examples."""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def function(mode="slice", *, rank=2, interpret=False, width=16, rows=2):
    padded = mode not in ("data", "data_no_other", "data_initialize", "data_bad", "empty")
    valid_width = width - 3 if padded else width
    shape = (rows, valid_width) if rank == 2 else (valid_width,)
    block = (rows, width) if rank == 2 else (width,)

    def kernel(x, y):
        lane = jnp.arange(width)
        mask = jnp.broadcast_to(lane < shape[-1], block)
        index = (slice(None),) * rank
        if mode == "slice":
            if rank == 2:
                y[:, : width // 2] = x[:, : width // 2] + 1
                y[:, width // 2 : valid_width] = x[:, width // 2 : valid_width] + 1
            else:
                y[: width // 2] = x[: width // 2] + 1
                y[width // 2 : valid_width] = x[width // 2 : valid_width] + 1
            return
        if mode.startswith("data"):
            original = x[...]
            chosen = original > 0
            if mode == "data_initialize":
                pp.store(y, index, original + 3, mask=chosen)
                loaded = pp.load(y, index, mask=chosen, other=-4)
                y[...] = loaded
                return
            if mode == "data":
                y[...] = original
                pp.store(y, index, original + 3, mask=chosen)
            else:
                loaded = pp.load(x, index, mask=chosen)
                y[...] = jnp.where(
                    chosen if mode == "data_no_other" else original >= 0, loaded + 3, -4
                )
            return
        if mode == "empty":
            y[...] = x[...]
            pp.store(y, index, jnp.ones(block, jnp.float32), mask=jnp.zeros(block, jnp.bool_))
            loaded = pp.load(y, index, mask=jnp.zeros(block, jnp.bool_), other=7)
            y[...] += loaded
            return
        if mode in ("read_where", "read_bad"):
            loaded = x[...]
            clipped = jnp.where(
                mask if mode == "read_where" else jnp.ones(block, jnp.bool_), loaded, 0
            )
            # Reading all staging lanes is allowed, but their undefined values
            # cannot contribute to a reduction over the complete block.
            y[...] = jnp.broadcast_to(jnp.sum(clipped), block)
            return
        holes = jnp.broadcast_to((lane < width // 4) | (lane >= width // 2), block) & mask
        if mode == "holes":
            y[...] = jnp.full(block, -5, jnp.float32)
            pp.store(y, index, pp.load(x, index, mask=holes, other=0) + 3, mask=holes)
        elif mode == "cover":
            pp.store(y, index, pp.load(x, index, mask=holes, other=0) + 3, mask=holes)
            pp.store(
                y, index, pp.load(x, index, mask=mask & ~holes, other=0) + 3, mask=mask & ~holes
            )
        elif mode == "missing":
            pp.store(y, index, pp.load(x, index, mask=holes, other=0), mask=holes)
        else:
            loaded = pp.load(x, index, mask=holes)
            chosen = holes if mode == "no_other" else mask
            y[...] = jnp.where(chosen, loaded + 3, -5)

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(1,),
        in_specs=(pl.BlockSpec(block, lambda _: (0,) * rank),),
        out_specs=pl.BlockSpec(block, lambda _: (0,) * rank),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )
    return result, shape


def expected(x, mode):
    x = np.asarray(x)
    holes = (np.arange(x.shape[-1]) < 4) | (np.arange(x.shape[-1]) >= 8)
    if mode in ("slice", "cover"):
        return x + (1 if mode == "slice" else 3)
    if mode in ("holes", "no_other"):
        return np.where(holes, x + 3, -5)
    if mode == "read_where":
        return np.full_like(x, x.sum())
    if mode == "data":
        return np.where(x > 0, x + 3, x)
    if mode in ("data_no_other", "data_initialize"):
        return np.where(x > 0, x + 3, -4)
    if mode == "empty":
        return x + 7
    raise ValueError(mode)


def dma_function(mode="disjoint", *, interpret=False):
    def kernel(x, y, sem):
        copy = tpu.make_async_copy(x.at[:, :128], y.at[:, :128], sem)
        copy.start()
        mask = jnp.broadcast_to(jnp.arange(256) >= (64 if mode == "overlap" else 128), (8, 256))
        pp.store(y, (slice(None), slice(None)), x[...] + 2, mask=mask)
        if mode != "missing_wait":
            copy.wait()

    spec = pl.BlockSpec((8, 256), lambda _: (0, 0))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 256), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.DMA,),
        compiler_params=tpu.CompilerParams(),
        interpret=interpret,
    )
