"""Native synchronous Buffered(revisit=ANY) kernels and independent references."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

WIDTH = 256


def function(
    *,
    mode="normal",
    pipeline_mode="any",
    programs=3,
    consecutive=1,
    receive_first=False,
    multiple=False,
    mesh=None,
):
    if pipeline_mode == "any":
        pipeline_mode = pl.Buffered(1, revisit=pl.RevisitMode.ANY)
    elif pipeline_mode == "immediate":
        pipeline_mode = pl.Buffered(1, revisit=pl.RevisitMode.IMMEDIATE)

    def window(p):
        return (p // consecutive + (mode == "wrong_block")) % 2

    def kernel(x, y, *resources):
        total, temporary, send, receive, ack = resources if multiple else (None, *resources)
        p = pl.program_id(0)
        peer = jax.lax.axis_index("d") if mode == "self_target" else 1 - jax.lax.axis_index("d")
        if mode != "no_ack":

            @pl.when(p > 0)
            def synchronize_staging():
                pl.semaphore_signal(ack, 1, device_id=(peer,))
                pl.semaphore_wait(ack, 1)

        output = y
        if mode == "initial_read":
            output[:] += 1
            pl.semaphore_signal(ack, 1, device_id=(peer,))
            pl.semaphore_wait(ack, 1)
        accumulate = mode in ("accumulate", "missing_init")
        if accumulate and mode != "missing_init":

            @pl.when((p < 2 * consecutive) & (p % consecutive == 0))
            def initialize():
                output[:] = jnp.zeros((WIDTH,), jnp.float32)

        def copy(width):
            target = temporary.at[:width] if accumulate else output.at[:width]
            dma = tpu.async_remote_copy(x.at[:width], target, send, receive, (peer,))
            if receive_first:
                dma.wait_recv()
            if mode != "missing_send":
                dma.wait_send()
            if not receive_first and mode != "missing_receive":
                dma.wait_recv()

        if mode in ("partial", "partial_first"):
            jax.lax.cond(
                p == (programs - 1 if mode == "partial" else 0),
                lambda: copy(WIDTH // 2),
                lambda: copy(WIDTH),
            )
        else:
            copy(WIDTH)
        if accumulate:
            output[:] += temporary[:]
        if multiple:

            @pl.when(p == 0)
            def initialize_total():
                total[:] = jnp.zeros((WIDTH,), jnp.float32)

            total[:] += x[:]

    output = pl.BlockSpec(
        (WIDTH,),
        lambda p: (window(p),),
        pipeline_mode=pipeline_mode,
    )
    output_shape = jax.ShapeDtypeStruct((2 * WIDTH,), jnp.float32)
    call = pl.pallas_call(
        kernel,
        out_shape=(output_shape, jax.ShapeDtypeStruct((WIDTH,), jnp.float32))
        if multiple
        else output_shape,
        grid=(programs,),
        in_specs=[pl.BlockSpec((WIDTH,), lambda p: (p,))],
        out_specs=(output, pl.BlockSpec((WIDTH,), lambda p: (0,))) if multiple else output,
        scratch_shapes=[
            tpu.VMEM((WIDTH,), jnp.float32),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR,
        ],
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)) if mesh is None else mesh
    return jax.shard_map(
        call,
        mesh=mesh,
        in_specs=P("d"),
        out_specs=(P("d"), P("d")) if multiple else P("d"),
        check_vma=False,
    )


def graph(**options):
    return jax.make_jaxpr(function(**options))(
        jax.ShapeDtypeStruct((2 * options.get("programs", 3) * WIDTH,), jnp.float32)
    )


def expected(x, *, mode="normal", programs=3, consecutive=1, multiple=False, **unused):
    source = np.asarray(x).reshape(2, programs, WIDTH)
    received = source if mode == "self_target" else source[::-1]
    output = np.zeros((2, 2, WIDTH), dtype=np.float32)
    for p in range(programs):
        window = (p // consecutive + (mode == "wrong_block")) % 2
        if mode == "accumulate":
            output[:, window] += received[:, p]
        elif mode == "partial" and p == programs - 1:
            output[:, window, : WIDTH // 2] = received[:, p, : WIDTH // 2]
        else:
            output[:, window] = received[:, p]
    result = output.reshape(-1)
    return (result, source.sum(axis=1).reshape(-1)) if multiple else result


def run_cpu(mode):
    """Record the locked TPU interpreter's missing ANY-policy implementation."""
    import json

    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    x = np.arange(6 * WIDTH, dtype=np.float32)
    with tpu.force_tpu_interpret_mode(params):
        executable = jax.jit(function()).lower(x).compile()
    print("READY", flush=True)
    try:
        jax.block_until_ready(executable(x))
        jax.effects_barrier()
    except Exception as error:
        if "Revisited block" not in str(error):
            raise
    else:
        raise AssertionError("locked interpreter unexpectedly implements ANY revisits")
    print(
        "RESULT "
        + json.dumps(dict(case="buffered_any", mode=mode, status="interpreter_rejects_revisit")),
        flush=True,
    )
