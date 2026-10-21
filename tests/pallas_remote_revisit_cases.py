"""Source kernels for ordered A/B/A remote writes, with no implicit output reload."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

WIDTH = 256


def function(
    *, explicit=True, mode="normal", receive_first=False, multiple=False, revisit=True, mesh=None
):
    def kernel(x, y, *resources):
        total, send, receive, ack = resources if multiple else (None, *resources)
        program = pl.program_id(0)
        peer = jax.lax.axis_index("d") if mode == "self_target" else 1 - jax.lax.axis_index("d")
        if mode != "no_ack":

            @pl.when(program > 0)
            def completed_previous_window():
                pl.semaphore_signal(ack, 1, device_id=(peer,))
                pl.semaphore_wait(ack, 1)

        window = 0 if mode == "missing_block" else (program + (mode == "wrong_block")) % 2
        if mode == "stale_accumulate":

            @pl.when(program == 2)
            def reuse_previous_output():
                y[:] = y[:] + 1

            # Order the stale read before either participant starts the next DMA.
            pl.semaphore_signal(ack, 1, device_id=(peer,))
            pl.semaphore_wait(ack, 1)

        def copy(width):
            destination = y.at[pl.ds(window * WIDTH, width)] if explicit else y.at[:width]
            transfer = tpu.async_remote_copy(x.at[:width], destination, send, receive, (peer,))
            if receive_first:
                transfer.wait_recv()
            if mode != "missing_send":
                transfer.wait_send()
            if not receive_first and mode != "missing_receive":
                transfer.wait_recv()

        if mode == "partial_revisit":
            jax.lax.cond(program == 2, lambda: copy(WIDTH // 2), lambda: copy(WIDTH))
        else:
            copy(WIDTH)
        if multiple:

            @pl.when(program == 0)
            def initialize_total():
                total[:] = jnp.zeros((WIDTH,), jnp.float32)

            total[:] += x[:]

    out_spec = (
        pl.BlockSpec((2 * WIDTH,), lambda p: (0,), memory_space=pl.ANY)
        if explicit
        else pl.BlockSpec(
            (WIDTH,), lambda p: ((p + (mode == "wrong_block")) % 2 if revisit else p,)
        )
    )
    out_shape = jax.ShapeDtypeStruct(((2 if explicit or revisit else 3) * WIDTH,), jnp.float32)
    call = pl.pallas_call(
        kernel,
        out_shape=(
            (
                out_shape,
                jax.ShapeDtypeStruct((WIDTH,), jnp.float32),
            )
            if multiple
            else out_shape
        ),
        grid=(3,),
        in_specs=[pl.BlockSpec((WIDTH,), lambda p: (p,))],
        out_specs=(out_spec, pl.BlockSpec((WIDTH,), lambda p: (0,))) if multiple else out_spec,
        scratch_shapes=[tpu.SemaphoreType.DMA, tpu.SemaphoreType.DMA, tpu.SemaphoreType.REGULAR],
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
    return jax.make_jaxpr(function(**options))(jax.ShapeDtypeStruct((6 * WIDTH,), jnp.float32))


def expected(x, *, mode="normal"):
    received = np.asarray(x).reshape(2, 3, WIDTH)
    if mode != "self_target":
        received = received[::-1]
    result = received[:, [2, 1]].copy()
    if mode == "partial_revisit":
        result[:, 0, WIDTH // 2 :] = received[:, 0, WIDTH // 2 :]
    if mode == "wrong_block":
        result = result[:, ::-1]
    return result.reshape(-1)


def run_cpu(case, dma_mode):
    """Official CPU oracle, including the interpreter's unconditional revisit rejection."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    explicit = case != "automatic"
    mode = (
        "partial_revisit" if case == "partial" else "wrong_block" if case == "wrong" else "normal"
    )
    params = tpu.InterpretParams(
        dma_execution_mode=dma_mode,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    x = np.arange(6 * WIDTH, dtype=np.float32)
    with tpu.force_tpu_interpret_mode(params):
        executable = (
            jax.jit(function(explicit=explicit, mode=mode, multiple=case == "multi"))
            .lower(x)
            .compile()
        )
    print("READY", flush=True)
    try:
        output = jax.block_until_ready(executable(x))
        jax.effects_barrier()
    except Exception as error:
        if explicit or "Revisited block" not in str(error):
            raise
        result = dict(case=case, mode=dma_mode, status="interpreter_rejects_revisit")
    else:
        if not explicit:
            raise AssertionError("pinned interpreter unexpectedly admitted automatic revisit")
        reference = expected(x, mode=mode)
        if case == "multi":
            reference = (reference, x.reshape(2, 3, WIDTH).sum(axis=1).reshape(-1))
        jax.tree.map(np.testing.assert_array_equal, output, reference)
        assert interpreter.races is not None and not interpreter.races.races_found
        result = dict(case=case, mode=dma_mode, status="ok")
    print("RESULT " + json.dumps(result), flush=True)
