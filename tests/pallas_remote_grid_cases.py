"""Actual Pallas sources for multidimensional remote DMA program grids."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

ROWS = 2
COLUMNS = 3
WIDTH = 128
LOCAL_SIZE = ROWS * COLUMNS * WIDTH


def binding(*, flat=False, three_dimensional=False, transpose_grid=False, **unused):
    from jax_reorder import CoreExecutionBinding

    grid = (
        (ROWS * COLUMNS,)
        if flat
        else (ROWS, 1, COLUMNS)
        if three_dimensional
        else (COLUMNS, ROWS)
        if transpose_grid
        else (ROWS, COLUMNS)
    )
    return CoreExecutionBinding((tuple(product(*(range(n) for n in grid))),))


def function(
    *,
    flat=False,
    parallel=False,
    all_parallel=False,
    three_dimensional=False,
    transpose_grid=False,
    deferred=False,
    explicit=False,
    receive_first=False,
    mode="normal",
    mesh=None,
):
    """Bank actual scratch/sem storage; no program-private allocation is assumed."""
    parallel = parallel or all_parallel
    if flat:
        grid = (ROWS * COLUMNS,)
        semantics = ("arbitrary",)
    elif three_dimensional:
        grid = (ROWS, 1, COLUMNS)
        semantics = (
            "parallel" if parallel else "arbitrary",
            "parallel" if all_parallel else "arbitrary",
            "parallel" if all_parallel else "arbitrary",
        )
    elif transpose_grid:
        grid = (COLUMNS, ROWS)
        semantics = (
            "parallel" if all_parallel else "arbitrary",
            "parallel" if parallel else "arbitrary",
        )
    else:
        grid = (ROWS, COLUMNS)
        semantics = (
            "parallel" if parallel else "arbitrary",
            "parallel" if all_parallel else "arbitrary",
        )
    explicit = explicit or parallel or deferred
    banks = ROWS * COLUMNS if all_parallel or deferred else ROWS

    def tile(*coordinates):
        if flat:
            row, column = coordinates[0] // COLUMNS, coordinates[0] % COLUMNS
        elif three_dimensional:
            row, singleton, column = coordinates
            column += singleton
        elif transpose_grid:
            column, row = coordinates
        else:
            row, column = coordinates
        return row * COLUMNS + column

    def input_mapping(*coordinates):
        index = tile(*coordinates)
        return ((index + (mode == "wrong_index_map")) % (ROWS * COLUMNS),)

    def kernel(x, y, temporary, local, send, receive, ack):
        if flat:
            row, column = pl.program_id(0) // COLUMNS, pl.program_id(0) % COLUMNS
            columns = COLUMNS
        elif three_dimensional:
            row, column = pl.program_id(0), pl.program_id(1) + pl.program_id(2)
            columns = pl.num_programs(1) * pl.num_programs(2)
        elif transpose_grid:
            column, row = pl.program_id(0), pl.program_id(1)
            columns = pl.num_programs(0)
        else:
            row, column = pl.program_id(0), pl.program_id(1)
            columns = pl.num_programs(1)
        index = row * columns + column
        owner = index if all_parallel else row
        slot = index if deferred else owner
        peer = jax.lax.axis_index("d")
        if mode != "self_target":
            peer = 1 - peer
        # A peer can start its next receive before the previous local source is reused.
        pl.semaphore_signal(ack.at[owner], 1, device_id=(peer,))
        pl.semaphore_wait(ack.at[owner], 1)
        if explicit:
            source_index = (index + (mode == "wrong_index_map")) % (ROWS * COLUMNS)
            bank = 0 if mode == "shared_scratch" else slot
            source = temporary.at[bank]
            tpu.async_copy(x.at[pl.ds(source_index * WIDTH, WIDTH)], source, local.at[slot]).wait()
        else:
            source = x
        target_index = column if mode == "overlapping_output" else index
        transfer = tpu.async_remote_copy(
            source,
            y.at[pl.ds(target_index * WIDTH, WIDTH)],
            send.at[slot],
            receive.at[slot],
            (peer,),
        )

        def complete(copy):
            if receive_first and mode != "missing_receive":
                copy.wait_recv()
            if mode != "missing_send":
                copy.wait_send()
            if not receive_first and mode != "missing_receive":
                copy.wait_recv()

        if deferred:

            @pl.when(column == columns - 1)
            def complete_previous_programs():
                for previous_column in range(COLUMNS):
                    previous = row * columns + previous_column
                    complete(
                        tpu.make_async_remote_copy(
                            temporary.at[previous],
                            y.at[pl.ds(previous * WIDTH, WIDTH)],
                            send.at[previous],
                            receive.at[previous],
                            (peer,),
                        )
                    )

        else:
            complete(transfer)

    full_spec = pl.BlockSpec((LOCAL_SIZE,), lambda *coordinates: (0,), memory_space=pl.ANY)
    output_size = COLUMNS * WIDTH if mode == "overlapping_output" else LOCAL_SIZE
    output_spec = pl.BlockSpec((output_size,), lambda *coordinates: (0,), memory_space=pl.ANY)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((output_size,), jnp.float32),
        grid=grid,
        in_specs=(full_spec if explicit else pl.BlockSpec((WIDTH,), input_mapping),),
        out_specs=output_spec,
        scratch_shapes=(
            tpu.VMEM((banks, WIDTH), jnp.float32),
            tpu.SemaphoreType.DMA((banks,)),
            tpu.SemaphoreType.DMA((banks,)),
            tpu.SemaphoreType.DMA((banks,)),
            tpu.SemaphoreType.REGULAR((banks,)),
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=semantics),
    )
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)) if mesh is None else mesh
    assert mesh.size == 2
    return jax.shard_map(call, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)


def graph(**options):
    return jax.make_jaxpr(function(**options))(jax.ShapeDtypeStruct((2 * LOCAL_SIZE,), jnp.float32))


def expected(x, *, mode="normal", **unused):
    shards = np.asarray(x).reshape(2, ROWS * COLUMNS, WIDTH)
    if mode != "self_target":
        shards = shards[::-1]
    if mode == "wrong_index_map":
        shards = np.roll(shards, -1, axis=1)
    return shards.reshape(-1)


def run_cpu(case, execution):
    """Isolated official TPU interpreter checks for the remote grid source kernels."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    options = {
        "serial": {},
        "flat": dict(flat=True),
        "serial_transposed": dict(transpose_grid=True),
        "three_dimensional": dict(three_dimensional=True),
        "parallel": dict(parallel=True),
        "all_parallel": dict(all_parallel=True),
        "parallel_transposed": dict(parallel=True, transpose_grid=True),
        "wrong_index_map": dict(mode="wrong_index_map"),
        "self_target": dict(parallel=True, mode="self_target"),
        "deferred": dict(parallel=True, deferred=True),
    }[case]
    x = np.arange(2 * LOCAL_SIZE, dtype=np.float32)
    params = tpu.InterpretParams(
        dma_execution_mode=execution,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    try:
        with tpu.force_tpu_interpret_mode(params):
            executable = jax.jit(function(**options)).lower(x).compile()
    except ValueError as error:
        if case != "parallel_transposed" or (
            "Dimensions with parallel semantics must form a prefix of the grid." not in str(error)
        ):
            raise
        print("READY", flush=True)
        print(
            "RESULT "
            + json.dumps(dict(case=case, mode=execution, status="interpreter_grid_limitation")),
            flush=True,
        )
        return
    print("READY", flush=True)
    result = jax.block_until_ready(executable(x))
    jax.effects_barrier()
    np.testing.assert_array_equal(result, expected(x, **options))
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, mode=execution, status="ok")), flush=True)
