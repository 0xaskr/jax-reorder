"""Concrete serial expansion cases, with independent NumPy and CPU checks."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def grouped(*, group_size=2, groups=2, split=False, mode="base"):
    steps = 2 * group_size

    def body(x, y, total):
        j = pl.program_id(1)
        count = pl.num_programs(1)
        if split:
            j = j * group_size + pl.program_id(2)
            count *= pl.num_programs(2)

        if mode != "missing_init":

            @pl.when(j % group_size == 0)
            def initialize():
                total[...] = jnp.zeros(total.shape, jnp.float32)

        value = x[...]
        y[...] = value * 2 + j + count
        if mode == "reduce":
            value = jnp.sum(value, axis=1, keepdims=True)
        total[...] = value if mode == "overwrite" else total[...] + value

    def position(*ids):
        return ids[0], ids[1] * group_size + ids[2] if split else ids[1]

    def summary(*ids):
        p, j = position(*ids)
        block = j // group_size
        if mode == "wrong_map":
            block = 1 - block
        elif mode == "revisit":
            block = j % 2
        elif mode == "gap":
            block = 0
        return p, block

    width = 1 if mode == "reduce" else 128
    grid = (groups, 2, group_size) if split else (groups, steps)
    full = pl.BlockSpec((8, 128), position)
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((groups * 8, steps * 128), jnp.float32),
            jax.ShapeDtypeStruct((groups * 8, 2 * width), jnp.float32),
        ),
        grid=grid,
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, width), summary)),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel",) + ("arbitrary",) * (len(grid) - 1)
        ),
    )


def inputs(*, group_size=2, groups=2, seed=7):
    return (
        np.random.default_rng(seed)
        .integers(-4, 5, size=(groups * 8, 2 * group_size * 128))
        .astype(np.float32),
    )


def reference(args, *, group_size=2, mode="base"):
    (x,) = args
    steps = 2 * group_size
    tiles = x.reshape(x.shape[0], steps, 128)
    y = (tiles * 2 + np.arange(steps, dtype=np.float32)[None, :, None] + steps).reshape(x.shape)
    total = tiles.reshape(x.shape[0], 2, group_size, 128).sum(axis=2)
    if mode == "reduce":
        total = total.sum(axis=2, keepdims=True)
    return y, total.reshape(x.shape[0], -1)


def grouped_loop(group_size=2):
    def body(x, y, total):
        total[...] = jnp.zeros((8, 128), jnp.float32)

        def step(k, _):
            value = x[:, pl.ds(k * 128, 128)]
            j = pl.program_id(1) * group_size + k
            y[:, pl.ds(k * 128, 128)] = value * 2 + j + 2 * group_size
            total[...] += value

        jax.lax.fori_loop(0, group_size, step, None)

    full = pl.BlockSpec((8, group_size * 128), lambda p, g: (p, g))
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((16, 256 * group_size), jnp.float32),
            jax.ShapeDtypeStruct((16, 256), jnp.float32),
        ),
        grid=(2, 2),
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 128), lambda p, g: (p, g))),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def grouped_dma(mode="base"):
    def body(x, y, total, scratch, sem):
        j = pl.program_id(0)

        @pl.when(j % 2 == 0)
        def start():
            total[...] = x[...]
            y[...] = x[...]
            source = x if mode == "input_source" else total
            tpu.make_async_copy(source, scratch, sem).start()

        @pl.when(j % 2 == 1)
        def finish():
            source = x if mode == "input_source" else total
            if mode == "overwrite_source":
                total[...] = x[...]
            if mode != "missing_wait":
                tpu.make_async_copy(source, scratch, sem).wait()
            y[...] = scratch[...]
            total[...] += x[...]

    full = pl.BlockSpec((8, 128), lambda j: (0, j))
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((8, 512), jnp.float32),
            jax.ShapeDtypeStruct((8, 256), jnp.float32),
        ),
        grid=(4,),
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 128), lambda j: (0, j // 2))),
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.DMA],
    )


def counter(mode="base"):
    def body(x, y, sem):
        j = pl.program_id(0)

        @pl.when(j % 2 == 0)
        def first():
            pl.semaphore_signal(sem, 1)

        @pl.when(j % 2 == 1)
        def second():
            pl.semaphore_wait(sem, 2 if mode == "wrong_count" else 1)

        y[...] = x[...] * 2

    full = pl.BlockSpec((8, 128), lambda j: (0, j))
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8, 512), jnp.float32),
        grid=(4,),
        in_specs=[full],
        out_specs=full,
        scratch_shapes=[tpu.SemaphoreType.REGULAR],
    )


def parallel_counter(*, core_index=None, rank_one=False, mode="base"):
    def body(x, y, sem):
        pl.semaphore_signal(sem, 1, core_index=core_index)
        pl.semaphore_wait(sem, 2 if mode == "wrong_count" else 1)
        if mode == "leftover":
            pl.semaphore_signal(sem, 1, core_index=core_index)
        y[...] = x[...] * 2

    shape = (512,) if rank_one else (8, 512)
    block = (128,) if rank_one else (8, 128)
    spec = pl.BlockSpec(block, (lambda j: (j,)) if rank_one else (lambda j: (0, j)))
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(4,),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[tpu.SemaphoreType.REGULAR],
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
    )


def dot_groups(groups=2):
    def body(x, w, y, total):
        j = pl.program_id(1)

        @pl.when(j % 2 == 0)
        def initialize():
            total[...] = jnp.zeros((8, 128), jnp.float32)

        value = jnp.dot(x[...], w[...], precision=jax.lax.Precision.HIGHEST)
        y[...] = value
        total[...] += value

    spec = pl.BlockSpec((8, 128), lambda p, j: (p, j))
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((groups * 8, 512), jnp.float32),
            jax.ShapeDtypeStruct((groups * 8, 256), jnp.float32),
        ),
        grid=(groups, 4),
        in_specs=[spec, pl.BlockSpec((128, 128), lambda p, j: (0, 0))],
        out_specs=(spec, pl.BlockSpec((8, 128), lambda p, j: (p, j // 2))),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def execute(case):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    size = 3 if case in ("group3", "loop3") else 2
    mode = "reduce" if case == "reduce" else "base"
    groups = 1 if case in ("dma", "counter", "parallel_counter") else 2
    if case == "dma":
        build = grouped_dma

        def expected(args):
            x = args[0].reshape(8, 4, 128)
            return (
                np.repeat(x[:, ::2], 2, axis=1).reshape(8, 512),
                (x[:, ::2] + x[:, 1::2]).reshape(8, 256),
            )

    elif case in ("counter", "parallel_counter"):
        build = parallel_counter if case == "parallel_counter" else counter

        def expected(args):
            return (args[0] * 2,)

    else:

        def build():
            return (
                grouped_loop(group_size=size)
                if case in ("loop2", "loop3")
                else grouped(group_size=size, mode=mode)
            )

        def expected(args):
            return reference(args, group_size=size, mode=mode)

    params = tpu.InterpretParams(
        detect_races=True,
        dma_execution_mode="on_wait",
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
    )
    with tpu.force_tpu_interpret_mode(params):
        compiled = jax.jit(build()).lower(*inputs(group_size=size, groups=groups)).compile()
    print("READY", flush=True)
    for seed in (7, 23):
        args = inputs(group_size=size, groups=groups, seed=seed)
        actual = jax.block_until_ready(compiled(*args))
        jax.effects_barrier()
        if case in ("counter", "parallel_counter"):
            actual = (actual,)
        for a, b in zip(actual, expected(args), strict=True):
            np.testing.assert_array_equal(a, b)
        assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, status="ok")), flush=True)
