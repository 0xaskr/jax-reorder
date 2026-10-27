"""JAX 0.11.2 squareplus and the original TPU scratch-accumulating matmul.

The baselines load complete pinned definitions, including their JIT decorators.
Pallas candidates use the unmodified upstream launcher with a separately written
kernel dependency; comparison always traces the native, interpret=False call.
"""

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import api
from jax._src import numpy as internal_jnp
from jax._src.numpy import util as numpy_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tests.upstream_tpu.sources import load_definitions

REVISION = "32544801e26115ac1794926d027148abf3baf009"


@functools.cache
def squareplus_original():
    return load_definitions(
        "jax",
        "nn",
        ("squareplus",),
        dict(api=api, jnp=internal_jnp, numpy_util=numpy_util),
    )["squareplus"]


def squareplus_half(x):
    """AI positive: distribute the final division across the two summands."""
    root = jnp.sqrt(jnp.square(x) + 4)
    return x * 0.5 + root * 0.5


def squareplus_wrong_scale(x):
    """AI negative: use a third instead of a half, preserving shape and dtype."""
    return (x + jnp.sqrt(jnp.square(x) + 4)) / 3


def squareplus_samples(seed):
    rng = np.random.default_rng(seed)
    return np.concatenate((np.array([-8, -1, 0, 1, 8]), rng.normal(size=11))).astype(np.float32)


def squareplus_reference(x):
    x = np.asarray(x, np.float64)
    return (x + np.sqrt(x * x + 4)) / 2


def matmul_reordered(x_ref, y_ref, out_ref, acc_ref):
    """AI positive: evaluate the independent tile product before the accumulator read."""

    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    product = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=acc_ref.dtype)
    acc_ref[...] = product + acc_ref[...]
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def matmul_overwrite(x_ref, y_ref, out_ref, acc_ref):
    """AI negative: discard preceding reduction tiles instead of accumulating them."""

    @pl.when(pl.program_id(2) == 0)
    def init():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] = jnp.dot(x_ref[...], y_ref[...], preferred_element_type=acc_ref.dtype)
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def matmul_missing_init(x_ref, y_ref, out_ref, acc_ref):
    """AI invalid variant: read scratch before its first initialization."""
    acc_ref[...] = acc_ref[...] + jnp.dot(
        x_ref[...], y_ref[...], preferred_element_type=acc_ref.dtype
    )
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


@functools.cache
def matmul_function(variant="original"):
    namespace = dict(functools=functools, jax=jax, jnp=jnp, pl=pl, pltpu=pltpu)
    if variant == "original":
        names = ("matmul_kernel", "matmul")
    else:
        namespace["matmul_kernel"] = {
            "reordered": matmul_reordered,
            "overwrite": matmul_overwrite,
            "missing_init": matmul_missing_init,
        }[variant]
        names = ("matmul",)
    launcher = load_definitions("jax", "matmul", names, namespace)["matmul"]
    return functools.partial(launcher, block_shape=(8, 128), block_k=128)


def matmul_samples(seed):
    rng = np.random.default_rng(seed)
    # Two reduction tiles are necessary to expose the overwrite counterexample;
    # the 2x2 parallel tile grid also retains the original shared scratch contract.
    return tuple(rng.normal(0, 0.25, shape).astype(np.float32) for shape in ((16, 256), (256, 256)))


def matmul_signatures():
    return tuple(jax.ShapeDtypeStruct(shape, jnp.float32) for shape in ((16, 256), (256, 256)))


def matmul_reference(args):
    x, y = (np.asarray(value, np.float64) for value in args)
    return x @ y


def interpreter_evidence():
    """Official TPU interpreter, separate from native-source graph comparison."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = pltpu.InterpretParams(
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    variants = ("original", "reordered", "overwrite")
    with pltpu.force_tpu_interpret_mode(params):
        compiled = {
            variant: jax.jit(matmul_function(variant)).lower(*matmul_signatures()).compile()
            for variant in variants
        }
    print("READY", flush=True)
    for seed in (7, 23):
        args = matmul_samples(seed)
        expected = matmul_reference(args)
        for variant, function in compiled.items():
            actual = np.asarray(jax.block_until_ready(function(*args)))
            jax.effects_barrier()
            if variant == "overwrite":
                # This independent formula identifies the actual bug, rather than
                # merely accepting any mismatch with the intended full product.
                last_tile = np.asarray(args[0][:, 128:], np.float64) @ np.asarray(
                    args[1][128:, :], np.float64
                )
                np.testing.assert_allclose(actual, last_tile, rtol=2e-4, atol=2e-4)
                assert not np.allclose(actual, expected, rtol=2e-4, atol=2e-4)
            else:
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
            assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT " + json.dumps({"case": "jax_matmul", "status": "ok", "races": False}),
        flush=True,
    )
