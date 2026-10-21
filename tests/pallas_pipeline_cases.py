"""Pinned, complete upstream kernels plus isolated AST mutations of their emitter."""

import ast
import contextlib
import functools
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.mesh import AbstractDevice
from jax._src.pallas.mosaic import pipeline
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

FIXTURES = Path(__file__).parent / "fixtures/pallas_kernels"


def environment():
    return jax.sharding.use_abstract_mesh(
        jax.sharding.AbstractMesh((), (), abstract_device=AbstractDevice("TPU v5", 1, "tpu"))
    )


def verify_sources():
    for entry in json.loads((FIXTURES / "jax_pipeline_manifest.json").read_text())["files"]:
        data = (
            (FIXTURES / entry["fixture"]) if entry["fixture"] else Path(inspect.getfile(pipeline))
        ).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == entry["git_blob"]
        )


def kernel(family, buffers=2, synchronous=False):
    tree = ast.parse((FIXTURES / "jax_tpu_pipeline_test.py.txt").read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PallasCallMultipleBufferedPipelineTest"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "test_" + family
    )
    source = next(n for n in method.body if isinstance(n, ast.FunctionDef))
    if synchronous:
        calls = [
            n
            for n in ast.walk(source)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "emit_pipeline"
        ]
        assert len(calls) == 1
        calls[0].keywords.append(ast.keyword(arg="no_pipelining", value=ast.Constant(True)))
    module = ast.fix_missing_locations(ast.Module(body=[source], type_ignores=[]))
    namespace = dict(
        jax=jax,
        jnp=jnp,
        pl=pl,
        pltpu=pltpu,
        block_shape=(128, 128),
        in_buffer_count=buffers,
        x_buffer_count=buffers,
        y_buffer_count=buffers,
        out_buffer_count=2,
    )
    exec(compile(module, "<upstream JAX pipeline kernel>", "exec"), namespace)
    return namespace[source.name]


def function(family="copy", *, buffers=2, synchronous=False):
    typ = jax.ShapeDtypeStruct((512, 512), jnp.int32 if family == "copy" else jnp.float32)
    return pl.pallas_call(
        kernel(family, buffers, synchronous),
        out_shape=typ,
        in_specs=[pl.BlockSpec(memory_space=pl.ANY)] * (1 if family == "copy" else 2),
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
    )


@functools.cache
def mutation(mode):
    if mode == "early_wait":
        original = pipeline._emit_pipeline
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        body = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "loop_body"
        )
        scope = next(n for n in body.body if isinstance(n, ast.With))
        slots = {}
        for i, node in enumerate(scope.body):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                args = node.value.args
                if args and isinstance(args[0], ast.Attribute):
                    slots[args[0].attr] = i
        a, b = slots["copy_in"], slots["wait_in"]
        scope.body[a], scope.body[b] = scope.body[b], scope.body[a]
    elif mode == "missing_final_wait":
        original = pipeline.Scheduler.finalize
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))

        class RemoveFinal(ast.NodeTransformer):
            def visit_Expr(self, node):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "wait_out"
                ):
                    return ast.Pass()
                return self.generic_visit(node)

        tree = RemoveFinal().visit(tree)
    else:
        name = (
            "wait_out"
            if mode == "missing_output_wait"
            else "wait_in"
            if mode in ("missing_input_wait", "wrong_wait_slot", "different_wait_descriptor")
            else "copy_in"
        )
        original = getattr(pipeline.BufferedRef, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        method = tree.body[0]
        if mode in ("missing_input_wait", "missing_output_wait"):

            class RemoveWait(ast.NodeTransformer):
                def visit_Expr(self, node):
                    if (
                        isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Attribute)
                        and node.value.func.attr == "wait"
                    ):
                        return ast.Pass()
                    return self.generic_visit(node)

            tree = RemoveWait().visit(tree)
        elif mode == "different_wait_descriptor":
            call = next(
                n
                for n in ast.walk(method)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "_window_ref_at"
            )
            call.args[0] = ast.parse(
                "lax.rem(wait_slot + jnp.uint32(1), jnp.uint32(self.buffer_count))", mode="eval"
            ).body
        elif mode in ("wrong_wait_slot", "reuse_inflight_slot"):
            target = "wait_slot" if mode == "wrong_wait_slot" else "slot"
            assignment = next(
                n
                for n in method.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)
            )
            assignment.value = ast.parse(
                "lax.rem(self.current_wait_in_slot + jnp.uint32(1), jnp.uint32(self.buffer_count))"
                if mode == "wrong_wait_slot"
                else "jnp.uint32(0)",
                mode="eval",
            ).body
        elif mode == "wrong_block":
            method.body.insert(
                1,
                ast.parse(
                    "grid_indices = (grid_indices[0], "
                    "lax.rem(grid_indices[1] + jnp.int32(1), jnp.int32(4)))"
                ).body[0],
            )
        else:
            raise ValueError(mode)
    ast.fix_missing_locations(tree)
    namespace = dict(pipeline.__dict__)
    exec(compile(tree, f"<AI mutation of JAX pipeline: {mode}>", "exec"), namespace)
    return original.__name__, namespace[original.__name__]


@contextlib.contextmanager
def candidate(mode):
    if mode is None:
        yield
    else:
        name, fn = mutation(mode)
        target = (
            pipeline
            if mode == "early_wait"
            else pipeline.Scheduler
            if mode == "missing_final_wait"
            else pipeline.BufferedRef
        )
        with patch.object(target, name, fn):
            yield


def trace(family="copy", mode=None, *, buffers=2, synchronous=False):
    typ = jax.ShapeDtypeStruct((512, 512), jnp.int32 if family == "copy" else jnp.float32)
    with environment(), candidate(mode):
        return jax.make_jaxpr(function(family, buffers=buffers, synchronous=synchronous))(
            *([typ] * (1 if family == "copy" else 2))
        )


def inputs(family, seed):
    rng = np.random.default_rng(seed)
    if family == "copy":
        return (jnp.asarray(rng.integers(-1000, 1000, (512, 512)), jnp.int32),)
    return tuple(jnp.asarray(rng.normal(0, 0.1, (512, 512)), jnp.float32) for _ in range(2))


def serial_grid_function(*, staged=False, overwrite=False):
    """Small boundary probe, separate from the unchanged upstream baselines."""

    def body(x, y, buffers, sems, output_sem):
        i = pl.program_id(0)
        next_slot = jnp.int32(0) if overwrite else jax.lax.rem(i, jnp.int32(2))

        @pl.when(i < 3)
        def prefetch():
            source = x if staged else x.at[pl.ds(i * 8, 8), :]
            pltpu.async_copy(source, buffers.at[next_slot], sems.at[next_slot])

        @pl.when(i > 0)
        def compute_previous():
            previous = i - 1
            slot = jax.lax.rem(previous, jnp.int32(2))
            source = x if staged else x.at[pl.ds(previous * 8, 8), :]
            pltpu.make_async_copy(source, buffers.at[slot], sems.at[slot]).wait()
            buffers[slot, :, :] = buffers[slot, :, :] + 1
            pltpu.async_copy(buffers.at[slot], y.at[pl.ds(previous * 8, 8), :], output_sem).wait()

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((24, 128), jnp.float32),
        in_specs=[
            pl.BlockSpec((8, 128), lambda i: (i, 0))
            if staged
            else pl.BlockSpec(memory_space=pl.ANY)
        ],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        grid=(4,),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        scratch_shapes=(
            pltpu.VMEM((2, 8, 128), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA,
        ),
    )


def execute(family="copy", mode=None, buffers=2, synchronous=False):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    verify_sources()
    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    args = inputs(family, 7)
    with environment(), candidate(mode), pltpu.force_tpu_interpret_mode(params):
        compiled = (
            jax.jit(function(family, buffers=buffers, synchronous=synchronous))
            .lower(*args)
            .compile()
        )
    print("READY", flush=True)
    for seed in (7, 23):
        args = inputs(family, seed)
        actual = np.asarray(jax.block_until_ready(compiled(*args)))
        jax.effects_barrier()
        expected = (
            np.asarray(args[0]) if family == "copy" else np.asarray(args[0]) @ np.asarray(args[1])
        )
        if mode in ("wrong_block", "missing_input_wait"):
            assert not np.array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)
        assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT "
        + json.dumps(
            dict(
                case=family,
                mode=mode,
                buffers=buffers,
                synchronous=synchronous,
                status="ok",
                races=False,
            )
        ),
        flush=True,
    )


def execute_grid():
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    x = jnp.arange(32 * 128, dtype=jnp.float32).reshape(32, 128)
    with pltpu.force_tpu_interpret_mode(params):
        compiled = jax.jit(serial_grid_function()).lower(x).compile()
    print("READY", flush=True)
    actual = jax.block_until_ready(compiled(x))
    jax.effects_barrier()
    np.testing.assert_array_equal(actual, x[:24] + 1)
    assert interpreter.races is not None and not interpreter.races.races_found
    print('RESULT {"case": "serial_grid", "status": "ok", "races": false}', flush=True)
