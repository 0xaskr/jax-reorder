"""Unmodified, pinned upstream calls and separate state-update mutations.

The JAX modules are supplied by the locked dependency. Their bytes were checked
against the GitHub commit in the manifest. TPU interpretation changes the backend
under an official context manager, never the kernel body or its grid contract.
"""

import ast
import functools
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.experimental.pallas.ops.tpu import flash_attention, matmul

from examples.pallas_kernels import function as pinned_function
from examples.pallas_kernels import inputs as pinned_inputs
from examples.pallas_kernels import oracle as pinned_oracle

MANIFEST = Path(__file__).parent / "fixtures/pallas_kernels/jax_state_manifest.json"
FAMILIES = ("jax_matmul", "jax_attention", "rope", "quantized_rope", "attention")


def verify_sources():
    for entry in json.loads(MANIFEST.read_text())["files"]:
        module = importlib.import_module(entry["module"])
        data = Path(inspect.getfile(module)).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == entry["git_blob"]
        )


def inputs(family, seed=7):
    if family not in ("jax_matmul", "jax_attention"):
        return pinned_inputs(family, seed)
    rng = np.random.default_rng(seed)
    shapes = (
        ((16, 256), (256, 256))
        if family == "jax_matmul"
        else ((1, 1, 128, 128), (1, 1, 256, 128), (1, 1, 256, 128))
    )
    return tuple(jnp.asarray(rng.normal(0, 0.25, s), jnp.float32) for s in shapes)


def reference(family, args):
    if family == "jax_matmul":
        return args[0] @ args[1]
    if family == "jax_attention":
        q, k, v = args
        return jax.nn.softmax(q @ jnp.swapaxes(k, -1, -2), axis=-1) @ v
    return pinned_oracle(family, args)


@functools.cache
def matmul_candidate(mode):
    """Mutate actual upstream syntax; the original function is never replaced."""
    tree = ast.parse(inspect.getsource(matmul.matmul_kernel))
    kernel = tree.body[0]
    assert isinstance(kernel, ast.FunctionDef)
    initialize, update = kernel.body[:2]
    assert isinstance(initialize, ast.FunctionDef) and isinstance(update, ast.Assign)
    assert isinstance(update.value, ast.BinOp) and isinstance(update.value.op, ast.Add)
    if mode == "reset_each":
        initialize.decorator_list[0].args[0] = ast.Constant(True)
    elif mode == "overwrite":
        update.value = update.value.right
    elif mode == "double_state":
        update.value.left = ast.BinOp(update.value.left, ast.Mult(), ast.Constant(2.0))
    elif mode == "missing_init":
        kernel.body.remove(initialize)
    else:
        raise ValueError(mode)
    ast.fix_missing_locations(tree)
    namespace = {"pl": pl, "jnp": jnp}
    exec(compile(tree, f"<AI mutation of JAX matmul: {mode}>", "exec"), namespace)
    return namespace["matmul_kernel"]


def function(family, mode=None, *, block_k=128):
    if family == "jax_matmul":
        if mode is None:
            return functools.partial(matmul.matmul, block_shape=(8, 128), block_k=block_k)

        def candidate(x, y):
            return pl.pallas_call(
                matmul_candidate(mode),
                out_shape=jax.ShapeDtypeStruct((x.shape[0], y.shape[1]), x.dtype),
                grid_spec=tpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    grid=(x.shape[0] // 8, y.shape[1] // 128, x.shape[1] // block_k),
                    in_specs=[
                        pl.BlockSpec((8, block_k), lambda i, _, k: (i, k)),
                        pl.BlockSpec((block_k, 128), lambda _, j, k: (k, j)),
                    ],
                    out_specs=pl.BlockSpec((8, 128), lambda i, j, _: (i, j)),
                    scratch_shapes=[tpu.VMEM((8, 128), jnp.float32)],
                ),
                compiler_params=tpu.CompilerParams(
                    dimension_semantics=("parallel", "parallel", "arbitrary")
                ),
            )(x, y)

        return candidate
    if family == "jax_attention":
        assert mode is None
        return functools.partial(
            flash_attention.flash_attention,
            block_sizes=flash_attention.BlockSizes(128, block_k, 128, 1),
        )
    return pinned_function(family, mode)


def trace(family, mode=None, *, block_k=128):
    args = inputs(family)
    specs = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args)
    return jax.make_jaxpr(function(family, mode, block_k=block_k))(*specs)


def execute(family, mode=None, *, block_k=128):
    """Called in isolated workers; reports actual execution after compilation."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    verify_sources()
    args = inputs(family)
    params = tpu.InterpretParams(
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=2 if family == "jax_matmul" else 1,
    )
    with tpu.force_tpu_interpret_mode(params):
        f = jax.jit(function(family, mode, block_k=block_k))
        compiled = f.lower(
            *jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), args)
        ).compile()
    print("READY", flush=True)
    for seed in (7, 23):
        args = inputs(family, seed)
        actual = jax.block_until_ready(compiled(*args))
        jax.effects_barrier()
        expected = reference(family, args)
        if mode is None:
            for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                np.testing.assert_allclose(
                    a, b, rtol=2e-4, atol=1 if np.issubdtype(a.dtype, np.integer) else 2e-4
                )
        else:
            assert not np.allclose(actual, expected, rtol=2e-4, atol=2e-4), (family, mode, seed)
        assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT "
        + json.dumps(
            {"case": family, "mode": mode, "block_k": block_k, "status": "ok", "races": False}
        ),
        flush=True,
    )
