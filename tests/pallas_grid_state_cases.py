"""Pinned JAX scratch/grid DMA kernels and explicitly separate AI variants."""

import ast
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FIXTURES = Path(__file__).parent / "fixtures/pallas_kernels"


def verify_sources():
    for entry in json.loads((FIXTURES / "jax_grid_state_manifest.json").read_text())["files"]:
        data = (FIXTURES / entry["fixture"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == entry["git_blob"]
        )


def upstream_kernel(family, mode=None):
    tree = ast.parse((FIXTURES / "jax_tpu_state_test.py.txt").read_text())
    name = "test_hoisted_scratch_space" if family == "scratch" else "test_hbm_hbm_grid_dma"
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    source = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "kernel")
    if mode is not None:
        assert family == "scratch"
        initialize, output = source.body[1], source.body[-1]
        output.decorator_list[0].args[0] = ast.Constant(True)
        if mode != "history":
            # Two state slots, visited A, B, A, B; output blocks are a history
            # with unique owners, not an implicit restoration of old staging.
            source.body[1:1] = ast.parse("scratch_ref = scratch_ref.at[i % 2]").body
            initialize.decorator_list[0].args[0] = ast.parse("i < 2", mode="eval").body
        if mode == "reset_each":
            initialize.decorator_list[0].args[0] = ast.Constant(True)
        elif mode == "missing_init":
            source.body.remove(initialize)
        elif mode == "wrong_slot":
            # Writing bank 0 twice leaves bank 1 uninitialized.
            initialize.body = ast.parse("scratch_bank[0] = x_ref[...]").body
            source.body[1:1] = ast.parse("scratch_bank = scratch_ref").body
        elif mode not in ("history", "banks"):
            raise ValueError(mode)
    module = ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[]))
    namespace = dict(jnp=jnp, pl=pl, pltpu=pltpu)
    exec(compile(module, f"<JAX {family}; AI variant={mode}>", "exec"), namespace)
    return namespace["kernel"]


def function(family="scratch", mode=None):
    original = upstream_kernel(family, mode if family == "scratch" else None)
    if family == "scratch":
        banks = mode is not None and mode != "history"
        steps = 4 if banks else 3
        return pl.pallas_call(
            original,
            out_shape=jax.ShapeDtypeStruct((8 * steps if mode else 8, 128), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(steps,),
                in_specs=[pl.BlockSpec((8, 128), lambda i: (i % 2 if banks else 0, 0))],
                out_specs=pl.BlockSpec((8, 128), lambda i: (i if mode else 0, 0)),
                scratch_shapes=[pltpu.VMEM((2, 8, 128) if banks else (8, 128), jnp.float32)],
            ),
        )

    def revisit(x_ref, y_ref):
        i = pl.program_id(0)

        @pl.when(i < 2)
        def initialize():
            original(x_ref, y_ref)

        @pl.when(i >= 2)
        def update():
            slot = 3 - i if mode == "reverse" else i - 2

            def body(temp, sem):
                load = pltpu.async_copy(y_ref.at[pl.ds(slot, 1)], temp, sem)
                if mode != "missing_wait":
                    load.wait()
                temp[...] *= 2
                destination = 1 - slot if mode == "wrong_destination" else slot
                pltpu.async_copy(temp, y_ref.at[pl.ds(destination, 1)], sem).wait()

            pl.run_scoped(body, pltpu.VMEM((1, 8, 128), jnp.float32), pltpu.SemaphoreType.DMA)

    return pl.pallas_call(
        original if mode is None else revisit,
        out_shape=jax.ShapeDtypeStruct((2, 8, 128), jnp.float32),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        grid=(2 if mode is None else 4,),
    )


def inputs(family="scratch", mode=None, seed=7):
    shape = (2, 8, 128) if family == "dma" else (8, 128) if mode in (None, "history") else (16, 128)
    return (jnp.asarray(np.random.default_rng(seed).normal(size=shape), jnp.float32),)


def reference(family, mode, args):
    (x,) = args
    if family == "dma":
        return x if mode is None else x * 2
    if mode is None:
        return ((x + 1) + 1) + 1
    if mode == "history":
        return jnp.concatenate([x + 1, (x + 1) + 1, ((x + 1) + 1) + 1])
    return jnp.concatenate([x + 1, (x + 1) + 1])


def trace(family="scratch", mode=None):
    args = inputs(family, mode)
    return jax.make_jaxpr(function(family, mode))(
        *jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args)
    )


def execute(family, mode):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    verify_sources()
    params = pltpu.InterpretParams(
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
    )
    with pltpu.force_tpu_interpret_mode(params):
        compiled = jax.jit(function(family, mode)).lower(*inputs(family, mode)).compile()
    print("READY", flush=True)
    for seed in (7, 23):
        args = inputs(family, mode, seed)
        actual = jax.block_until_ready(compiled(*args))
        jax.effects_barrier()
        expected = reference(family, mode, args)
        if mode in ("reset_each", "wrong_destination"):
            assert not np.allclose(actual, expected)
        else:
            np.testing.assert_array_equal(actual, expected)
        assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(family=family, mode=mode, status="ok")), flush=True)
