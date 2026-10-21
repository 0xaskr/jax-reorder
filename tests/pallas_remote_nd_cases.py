"""Real matrix/tensor DMA sources with independent complete NumPy references."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def nd_function(
    *,
    shape=(16, 256),
    block_shape=None,
    partition_axis=0,
    mode="full",
    wrong=False,
    receive_first=False,
    self_target=False,
    mesh=None,
    interpret=False,
):
    mesh = mesh or Mesh(np.array(jax.devices("cpu")[:2]), ("d",))
    assert mesh.size == 2
    block_shape = block_shape or shape
    changed = next(
        (i for i, (a, b) in enumerate(zip(shape, block_shape, strict=True)) if a != b), 0
    )
    grid = shape[changed] // block_shape[changed]

    def mapping(p):
        return tuple(p if i == changed else 0 for i in range(len(shape)))

    spec = pl.BlockSpec(block_shape, mapping)

    def kernel(x, y, temp, send, recv, ack, local):
        device = lax.axis_index("d")
        peer = device if self_target else 1 - device
        if grid > 1:
            pl.semaphore_signal(ack, 1, device_id=(peer,))
            pl.semaphore_wait(ack, 1)
        if mode == "full":
            source, target = x, y
        elif mode in (
            "columns",
            "no_wait",
            "gap",
            "uninitialized",
            "inflight_hole",
            "inflight_overlap",
            "local_dma",
            "shared_dma_counter",
        ):
            # Initialize every output lane; transfer only one column tile.
            if mode in ("local_dma", "shared_dma_counter"):
                tpu.async_copy(x, temp, recv if mode == "shared_dma_counter" else local).wait()
            else:
                temp[...] = x[...]
            source = temp.at[..., 128:] if wrong else temp.at[..., :128]
            target = y.at[..., 128:]
            if mode != "gap":
                y[..., :128] = x[..., :128]
            if mode == "uninitialized":
                source = y.at[..., 128:]
        elif mode == "ref_access":
            temp.reshape((2, 8, 256))[...] = x.reshape((2, 8, 256))[...] + (1 if wrong else 0)
            source, target = temp, y
        elif mode == "transpose":
            source = x.transpose((1, 0, 2))
            target = y.reshape((shape[1], shape[0], shape[2]))
            if wrong:
                source = x.reshape((shape[1], shape[0], shape[2]))
        else:
            source = (
                x.reshape((2, 8, 256)).at[:, :, 128:]
                if wrong
                else x.reshape((2, 8, 256)).at[:, :, :128]
            )
            target = y.reshape((2, 8, 256)).at[:, :, 128:]
            y[:, :128] = x[:, :128]
        copy = tpu.async_remote_copy(source, target, send, recv, (peer,))
        if mode.startswith("inflight"):
            temp[..., 128:] = x[..., 128:] + 1 if mode == "inflight_hole" else x[..., 128:]
            if mode == "inflight_overlap":
                temp[..., :128] = x[..., :128]
        if mode == "no_wait":
            temp[...] = y[...]
        if receive_first:
            copy.wait_recv()
            copy.wait_send()
        else:
            copy.wait()

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(grid,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(
            tpu.VMEM(block_shape, jnp.float32),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR,
            tpu.SemaphoreType.DMA,
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )
    partition = (
        P(*(None if i != partition_axis else "d" for i in range(len(shape))))
        if partition_axis is not None
        else P()
    )
    return jax.shard_map(
        function, mesh=mesh, in_specs=partition, out_specs=partition, check_vma=False
    )


def expected(
    x, *, shape=(16, 256), partition_axis=0, mode="full", wrong=False, self_target=False, **unused
):
    shards = np.split(x, 2, axis=partition_axis) if partition_axis is not None else (x, x)
    outputs = []
    for device in range(2):
        peer = device if self_target else 1 - device
        if mode == "full":
            out = shards[peer].copy()
        elif mode == "ref_access":
            out = shards[peer] + (1 if wrong else 0)
        elif mode == "transpose":
            out = shards[peer].copy() if wrong else shards[peer].transpose(1, 0, 2).reshape(shape)
        else:
            out = shards[device].copy()
            out[..., 128:] = shards[peer][..., 128:] if wrong else shards[peer][..., :128]
        outputs.append(out)
    return (
        np.concatenate(outputs, axis=partition_axis) if partition_axis is not None else outputs[0]
    )


def run_cpu(case, execution):
    """Official CPU TPU-interpreter checks of actual multidimensional DMA sources."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    options = {
        "matrix": dict(shape=(16, 256), partition_axis=1),
        "tensor": dict(shape=(2, 8, 256), partition_axis=2),
        "columns": dict(shape=(16, 256), partition_axis=1, mode="columns"),
        "wrong_columns": dict(shape=(16, 256), partition_axis=1, mode="columns", wrong=True),
        "tensor_columns": dict(shape=(2, 8, 256), partition_axis=1, mode="columns"),
        "staging": dict(shape=(16, 256), block_shape=(16, 128), partition_axis=1),
        "hole": dict(shape=(16, 256), partition_axis=1, mode="inflight_hole"),
    }[case]
    shape = tuple(
        n * (2 if i == options["partition_axis"] else 1) for i, n in enumerate(options["shape"])
    )
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    params = tpu.InterpretParams(
        dma_execution_mode=execution,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        executable = jax.jit(nd_function(**options)).lower(x).compile()
    print("READY", flush=True)
    result = jax.block_until_ready(executable(x))
    jax.effects_barrier()
    np.testing.assert_array_equal(result, expected(x, **options))
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, mode=execution, status="ok")), flush=True)
