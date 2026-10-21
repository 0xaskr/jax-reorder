"""Pinned upstream ANY reduction and separate AI local-revisit variants."""

import ast

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from tests.pallas_grid_state_cases import FIXTURES, verify_sources

WIDTH = 256


def upstream_function():
    verify_sources()
    tree = ast.parse((FIXTURES / "jax_tpu_state_test.py.txt").read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_block_reduction_with_non_immediate_revisit"
    )
    body = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "body")
    namespace = dict(jnp=jnp, pl=pl)
    exec(
        compile(ast.Module(body=[body], type_ignores=[]), "<JAX pinned ANY reduction>", "exec"),
        namespace,
    )
    return pl.pallas_call(
        namespace["body"],
        out_shape=jax.ShapeDtypeStruct((8, 384), jnp.float32),
        grid=(3, 3),
        in_specs=[pl.BlockSpec((8, 128), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec(
            (8, 128),
            lambda i, j: (0, j),
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
        ),
    )


def function(mode="accumulate", *, steps=5, groups=2, consecutive=1, multiple=False, revisit="any"):
    pipeline = pl.Buffered(1, revisit=pl.RevisitMode(revisit)) if revisit else None

    def kernel(x, y, *others):
        j = pl.program_id(1)
        if mode != "missing_init":

            @pl.when(
                True if mode == "reset_each" else (j < 2 * consecutive) & (j % consecutive == 0)
            )
            def initialize():
                y[...] = jnp.zeros_like(y)

        if mode == "partial":

            @pl.when(j < steps - 1)
            def full():
                y[...] += x[...]

            @pl.when(j == steps - 1)
            def half():
                y[:, : WIDTH // 2] += x[:, : WIDTH // 2]
        elif mode == "overwrite":
            y[...] = x[...]
        else:
            y[...] += x[...]
        if multiple:
            total = others[0]

            @pl.when(j == 0)
            def initialize_total():
                total[...] = jnp.zeros_like(total)

            total[...] += x[...]

    output = pl.BlockSpec(
        (8, WIDTH),
        lambda i, j: (i, (j // consecutive + (mode == "wrong_map")) % 2),
        pipeline_mode=pipeline,
    )
    shape = jax.ShapeDtypeStruct((groups * 8, 2 * WIDTH), jnp.float32)
    return pl.pallas_call(
        kernel,
        out_shape=(shape, jax.ShapeDtypeStruct((groups * 8, WIDTH), jnp.float32))
        if multiple
        else shape,
        grid=(groups, steps),
        in_specs=[pl.BlockSpec((8, WIDTH), lambda i, j: (i, j))],
        out_specs=(output, pl.BlockSpec((8, WIDTH), lambda i, j: (i, 0))) if multiple else output,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def expected(x, mode="accumulate", *, consecutive=1, multiple=False):
    source = x.reshape(x.shape[0], -1, WIDTH)
    result = np.zeros((x.shape[0], 2, WIDTH), np.float32)
    for j in range(source.shape[1]):
        target = (j // consecutive + (mode == "wrong_map")) % 2
        if mode == "reset_each":
            result[:, target] = 0
        if mode == "partial" and j == source.shape[1] - 1:
            result[:, target, : WIDTH // 2] += source[:, j, : WIDTH // 2]
        elif mode == "overwrite":
            result[:, target] = source[:, j]
        else:
            result[:, target] += source[:, j]
    output = result.reshape(x.shape[0], -1)
    return (output, source.sum(axis=1)) if multiple else (output,)


def dma_function(mode="normal"):
    def kernel(x, y, sem):
        p = pl.program_id(0)

        @pl.when(p < 2)
        def initialize():
            y[...] = jnp.zeros_like(y)

        copy = tpu.async_copy(x, y, sem)
        if mode not in ("missing_wait", "write_pending"):
            copy.wait()
        if mode == "write_pending":
            y[...] = x[...]

    output = pl.BlockSpec(
        (8, WIDTH),
        lambda p: (0, p % 2),
        pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
    )
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 2 * WIDTH), jnp.float32),
        grid=(3,),
        in_specs=[pl.BlockSpec((8, WIDTH), lambda p: (0, p))],
        out_specs=output,
        scratch_shapes=[tpu.SemaphoreType.DMA],
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
