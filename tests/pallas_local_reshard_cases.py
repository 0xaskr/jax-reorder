"""Original jnp.where sources with Pallas-local manual sharding metadata."""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def where_function(*, masked=False, branch=False, mesh=None, fallback=7):
    shape = (8, 256)
    spec = pl.BlockSpec(shape, lambda _: (0, 0))

    def kernel(x, y, temporary, send, recv):
        tpu.async_remote_copy(x, temporary, send, recv, (1 - jax.lax.axis_index("d"),)).wait()
        mask = jnp.broadcast_to(jnp.arange(256)[None, :] < 192, shape)
        value = (
            pp.load(temporary, (slice(None), slice(None)), mask=mask) if masked else temporary[...]
        )
        if branch:
            value = jax.lax.cond(
                jax.lax.axis_index("d") == 0,
                lambda v: jnp.where(mask, v, fallback),
                lambda v: jnp.where(mask, v, fallback),
                value,
            )
        else:
            value = jnp.where(mask, value, fallback)
        y[...] = value

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.VMEM(shape, jnp.float32), tpu.SemaphoreType.DMA, tpu.SemaphoreType.DMA),
        compiler_params=tpu.CompilerParams(),
    )
    mesh = mesh or Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    return jax.shard_map(
        call, mesh=mesh, in_specs=P("d", None), out_specs=P("d", None), check_vma=False
    )


def run_cpu(branch, mode):
    """Official CPU execution of original jnp.where in a remote kernel."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    x = np.arange(4096, dtype=np.float32).reshape(16, 256)
    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        executable = jax.jit(where_function(branch=branch == "branch")).lower(x).compile()
    print("READY", flush=True)
    result = jax.block_until_ready(executable(x))
    jax.effects_barrier()
    expected = np.where(np.arange(256)[None, :] < 192, np.concatenate((x[8:], x[:8])), 7)
    np.testing.assert_array_equal(result, expected)
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=branch, mode=mode, status="ok")), flush=True)
