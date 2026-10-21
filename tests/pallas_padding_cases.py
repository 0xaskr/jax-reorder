"""Source-only padding kernels shared by CPU, projection and offline TPU probes."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def padded_reduction(*, kind="sum", limit=5, other=0, rank=2, width=8, interpret=False):
    block = (2, width) if rank == 2 else (width,)
    out_shape = (2, 1) if rank == 2 else (1,)

    def kernel(x, y):
        raw = x[...]
        clean = jnp.where(jnp.arange(width) < limit, raw, other)
        if kind in ("raw", "raw_add"):
            clean = raw + (0 if kind == "raw_add" else jnp.float32(0))
        elif kind == "cancel":
            clean = raw - raw
        elif kind == "zero":
            clean = raw * 0
        elif kind == "log":
            clean = jnp.log(jnp.where(jnp.arange(width) < limit, raw, 1))
        elif kind == "post_mask":
            clean = jnp.broadcast_to(jnp.sum(raw, axis=-1, keepdims=True), block)
            clean = jnp.where(jnp.arange(width) < limit, clean, 0)
        if kind == "mean":
            result = jnp.mean(clean, axis=-1, keepdims=True)
        elif kind == "max":
            result = jnp.max(clean, axis=-1, keepdims=True)
        else:
            result = jnp.sum(clean, axis=-1, keepdims=True)
        y[...] = result

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=(1,),
        in_specs=[pl.BlockSpec(block, lambda p: (0,) * rank)],
        out_specs=pl.BlockSpec(out_shape, lambda p: (0,) * rank),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )


def native_tail(*, kind="sum", other=0, limit=384, interpret=False):
    """Two 256-lane programs; the second contains 128 allocated padding lanes."""
    block = (8, 256)

    def kernel(x, y):
        mask = pl.program_id(0) * 256 + jnp.arange(256) < limit
        clean = jnp.where(mask, x[...], other)
        reduce = {"sum": jnp.sum, "mean": jnp.mean, "max": jnp.max}[kind]
        y[...] = jnp.broadcast_to(reduce(clean, axis=-1, keepdims=True), (8, 128))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 256), jnp.float32),
        grid=(2,),
        in_specs=[pl.BlockSpec(block, lambda p: (0, p))],
        out_specs=pl.BlockSpec((8, 128), lambda p: (0, p)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )


def padded_consumer(*, mode="dot"):
    """Selection-predicate and contraction propagation of the same partial load."""
    shape = (2, 2) if mode in ("dot", "raw_dot") else (2, 1)

    def kernel(x, y):
        raw = x[...]
        clean = jnp.where(jnp.arange(8) < 5, raw, 0)
        if mode == "raw_dot":
            clean = raw
        if mode in ("dot", "raw_dot"):
            y[...] = clean @ clean.T
        elif mode == "reverse_select":
            y[...] = jnp.sum(jnp.where(jnp.arange(8) >= 5, 0, raw), axis=1, keepdims=True)
        else:
            y[...] = jnp.sum(jnp.where(raw > 0, raw, 0), axis=1, keepdims=True)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        in_specs=[pl.BlockSpec((2, 8), lambda: (0, 0))],
        out_specs=pl.BlockSpec(shape, lambda: (0, 0)),
        compiler_params=tpu.CompilerParams(),
    )
