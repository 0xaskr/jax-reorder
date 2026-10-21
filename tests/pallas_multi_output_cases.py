"""AI multi-output cases; the upstream scratch body is kept unchanged."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from tests.pallas_grid_state_cases import upstream_kernel, verify_sources


def function(mode="base", *, summary_first=False, reverse=False, groups=2, steps=3):
    def body(x, *outputs):
        y, s = reversed(outputs) if summary_first else outputs
        j = pl.program_id(1)

        if mode != "missing_init":

            @pl.when(True if mode == "reset_each" else j == 0)
            def initialize():
                s[...] = jnp.zeros_like(x[...])

        value = x[...]
        s[...] = value if mode == "overwrite" else s[...] + value
        if mode == "missing_tile":

            @pl.when(j == steps - 1)
            def finish():
                y[...] = value * 2

        elif mode == "stale_tile":

            @pl.when(j == 0)
            def first():
                y[...] = value

            @pl.when(j > 0)
            def following():
                y[...] += value

        else:
            y[...] = s[...] if mode == "prefix" else value * 2

    shapes = [
        jax.ShapeDtypeStruct((groups * 8, steps * 128), jnp.float32),
        jax.ShapeDtypeStruct((groups * 8, 128), jnp.float32),
    ]
    specs = [
        pl.BlockSpec((8, 128), lambda i, j: (i, steps - 1 - j if reverse else j)),
        pl.BlockSpec((8, 128), lambda i, j: (groups - 1 - i if mode == "wrong_map" else i, 0)),
    ]
    if summary_first:
        shapes.reverse()
        specs.reverse()
    return pl.pallas_call(
        body,
        out_shape=tuple(shapes),
        grid=(groups, steps),
        in_specs=[pl.BlockSpec((8, 128), lambda i, j: (i, j))],
        out_specs=tuple(specs),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def inputs(seed=7, *, groups=2, steps=3):
    # Small integral floats make both sequential and NumPy reductions exact.
    return (
        np.random.default_rng(seed)
        .integers(-8, 9, size=(groups * 8, steps * 128))
        .astype(np.float32),
    )


def reference(args, *, prefix=False, reverse=False, summary_first=False):
    (x,) = args
    tiles = np.asarray(x).reshape(x.shape[0], -1, 128)
    y = np.cumsum(tiles, axis=1) if prefix else 2 * tiles
    if reverse:
        y = y[:, ::-1]
    outputs = (y.reshape(x.shape), np.sum(tiles, axis=1))
    return outputs[::-1] if summary_first else outputs


def trace(mode="base", **kwargs):
    args = inputs(groups=kwargs.get("groups", 2), steps=kwargs.get("steps", 3))
    return jax.make_jaxpr(function(mode, **kwargs))(
        *jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args)
    )


def nested_function():
    def body(x, y, subtotal, total):
        i, k = pl.program_id(1), pl.program_id(2)

        @pl.when(k == 0)
        def start_subtotal():
            subtotal[...] = jnp.zeros_like(x[...])

        @pl.when((i == 0) & (k == 0))
        def start_total():
            total[...] = jnp.zeros_like(x[...])

        y[...] = 2 * x[...]
        subtotal[...] += x[...]
        total[...] += x[...]

    full = pl.BlockSpec((8, 128), lambda p, i, k: (p, 3 * i + k))
    return pl.pallas_call(
        body,
        out_shape=tuple(
            jax.ShapeDtypeStruct(shape, jnp.float32) for shape in [(16, 768), (16, 256), (16, 128)]
        ),
        grid=(2, 2, 3),
        in_specs=[full],
        out_specs=(
            full,
            pl.BlockSpec((8, 128), lambda p, i, k: (p, i)),
            pl.BlockSpec((8, 128), lambda p, i, k: (p, 0)),
        ),
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary")
        ),
    )


def nested_reference(args):
    (x,) = args
    return (
        2 * x,
        x.reshape(16, 2, 3, 128).sum(axis=2).reshape(16, 256),
        x.reshape(16, 6, 128).sum(axis=1),
    )


def loop_function():
    def body(x, y, total):
        total[...] = jnp.zeros((8, 128), jnp.float32)

        def step(j, _):
            value = x[:, pl.ds(j * 128, 128)]
            y[:, pl.ds(j * 128, 128)] = value * 2
            total[...] += value

        jax.lax.fori_loop(0, 3, step, None)

    full = pl.BlockSpec((8, 384), lambda p: (p, 0))
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((16, 384), jnp.float32),
            jax.ShapeDtypeStruct((16, 128), jnp.float32),
        ),
        grid=(2,),
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 128), lambda p: (p, 0))),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
    )


def upstream_function():
    original = upstream_kernel("scratch")

    def body(x, history, final, scratch):
        original(x, final, scratch)
        history[...] = scratch[...]

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((24, 128), jnp.float32),
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
        ),
        grid=(3,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
        out_specs=(
            pl.BlockSpec((8, 128), lambda i: (i, 0)),
            pl.BlockSpec((8, 128), lambda i: (0, 0)),
        ),
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32)],
    )


def dma_function(mode="long_source"):
    def body(x, short, long, scratch, sem):
        i = pl.program_id(0)

        @pl.when(i == 0)
        def first():
            scratch[...] = x[...]
            if mode in ("long_source", "short_source"):
                long[...] = x[...]
                short[...] = x[...]
                source = short if mode == "short_source" else long
                tpu.make_async_copy(source, scratch, sem).start()
            else:
                short[...] = x[...]
                target = short if mode == "short_destination" else long
                tpu.make_async_copy(scratch, target, sem).start()

        @pl.when(i == 1)
        def second():
            if mode in ("long_source", "short_source"):
                source = short if mode == "short_source" else long
                tpu.make_async_copy(source, scratch, sem).wait()
                short[...] = scratch[...]
            else:
                target = short if mode == "short_destination" else long
                if mode != "missing_wait":
                    tpu.make_async_copy(scratch, target, sem).wait()
                short[...] = long[...]

    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((16, 128), jnp.float32),
            jax.ShapeDtypeStruct((8, 128), jnp.float32),
        ),
        grid=(2,),
        in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
        out_specs=(
            pl.BlockSpec((8, 128), lambda i: (i, 0)),
            pl.BlockSpec((8, 128), lambda i: (0, 0)),
        ),
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.DMA],
    )


def execute(case):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    groups, steps = 2, 3
    if case == "nested":
        build, expected = nested_function, nested_reference
        steps = 6
    elif case == "loop":
        build, expected = loop_function, reference
    elif case == "upstream":
        verify_sources()
        build = upstream_function
        groups, steps = 1, 1

        def expected(args):
            (x,) = args
            return np.concatenate([x + 1, x + 2, x + 3]), x + 3

    elif case.startswith("dma_"):

        def build():
            return dma_function(case.removeprefix("dma_"))

        groups, steps = 1, 1

        def expected(args):
            (x,) = args
            return np.concatenate([x, x]), x

    else:
        kwargs = dict(summary_first=case == "summary_first", reverse=case == "reverse")

        def build():
            return function(case, **kwargs)

        def expected(args):
            return reference(args, prefix=case == "prefix", **kwargs)

    params = tpu.InterpretParams(
        detect_races=True,
        dma_execution_mode="on_wait",
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
    )
    with tpu.force_tpu_interpret_mode(params):
        compiled = jax.jit(build()).lower(*inputs(groups=groups, steps=steps)).compile()
    print("READY", flush=True)
    for seed in (7, 23):
        args = inputs(seed, groups=groups, steps=steps)
        actual = jax.block_until_ready(compiled(*args))
        jax.effects_barrier()
        wanted = expected(args)
        if case in ("reset_each", "wrong_map"):
            np.testing.assert_array_equal(actual[0], wanted[0])
            assert not np.array_equal(actual[1], wanted[1])
        else:
            for a, b in zip(actual, wanted, strict=True):
                np.testing.assert_array_equal(a, b)
        assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, status="ok")), flush=True)
