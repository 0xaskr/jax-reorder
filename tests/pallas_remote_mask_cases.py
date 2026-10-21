"""Actual remote kernels with ordinary masked accesses and NumPy references."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def masked_function(mode="store", *, receive_first=False, mesh=None, identity_indices=False):
    shape = (8, 256)
    spec = pl.BlockSpec(shape, lambda _: (0, 0))

    def kernel(x, y, temp, send, recv):
        peer = 1 - lax.axis_index("d")
        cols = jnp.broadcast_to(jnp.arange(256)[None, :], shape)
        rows = jnp.broadcast_to(jnp.arange(8)[:, None], shape)
        index = (rows, cols) if identity_indices else (slice(None), slice(None))
        mask = cols < (128 if mode == "partial_init" else 192)
        if mode in ("disjoint", "overlap", "no_wait"):
            temp[...] = x[...]
            copy = tpu.async_remote_copy(temp.at[:, :128], y.at[:, :128], send, recv, (peer,))
            pp.store(
                temp,
                (slice(None), slice(None)),
                x[...] + 100,
                mask=cols >= (127 if mode == "overlap" else 128),
            )
            y[:, 128:] = temp[:, 128:]
            if mode != "no_wait":
                copy.wait()
            return
        copy = tpu.async_remote_copy(
            x, y if mode.startswith("empty") else temp, send, recv, (peer,)
        )
        if mode.startswith("empty"):
            empty = (
                (cols < 0) & (x[...] > 0) if mode == "empty_data" else jnp.zeros(shape, jnp.bool_)
            )
            ignored = pp.load(y, index, mask=empty, other=7)
            pp.store(y, index, ignored, mask=empty)
            if mode != "empty_no_wait":
                copy.wait()
            return
        if receive_first:
            copy.wait_recv()
            copy.wait_send()
        else:
            copy.wait()
        if mode in ("store", "holes", "data_store"):
            y[...] = temp[...]
            active = (cols < 32) | (cols >= 192) if mode == "holes" else mask
            if mode == "data_store":
                active = temp[...] > 2500
            pp.store(y, index, temp[...] + 100, mask=active)
        elif mode in ("swap", "swap_leak"):
            old = pp.swap(temp, index, temp[...] + 100, mask=mask)
            y[...] = old if mode == "swap_leak" else lax.select(mask, old, temp[...])
        elif mode == "vector_other":
            y[...] = pp.load(temp, index, mask=mask, other=x[...] + 100)
        elif mode == "transpose_store":
            y[...] = temp[...]
            active = jnp.broadcast_to(jnp.arange(256)[:, None] < 192, (256, 8))
            pp.store(y.transpose((1, 0)), None, temp[...].T + 100, mask=active)
        elif mode == "device_mask":
            active = cols < (128 + lax.axis_index("d") * 64)
            y[...] = pp.load(temp, index, mask=active, other=7)
        elif mode == "squeezed":
            y[...] = temp[...]
            y[0, :] = pp.load(
                temp, (jnp.int32(1), slice(None)), mask=jnp.arange(256) < 192, other=7
            )
        elif mode == "transpose":
            active = jnp.broadcast_to(jnp.arange(256)[:, None] < 192, (256, 8))
            value = pp.load(temp.transpose((1, 0)), None, mask=active, other=7)
            y[...] = value.T
        elif mode in ("partial_init", "uninitialized"):
            # The output starts uninitialized: the inactive half may be read only as other.
            pp.store(y, index, temp[...], mask=cols < 128)
            value = pp.load(
                y, index, mask=cols < (129 if mode == "uninitialized" else 128), other=7
            )
            y[...] = value
        else:
            active = temp[...] > 2500 if mode.startswith("data_") else mask
            other = 7 if mode in ("load", "data_load") else None
            value = pp.load(temp, index, mask=active, other=other)
            if mode in ("where", "data_where"):
                value = lax.select(active, value, jnp.full(shape, 7, jnp.float32))
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


def expected(x, mode="store"):
    x = np.asarray(x)
    shards = np.split(x, 2, axis=0)
    outputs = []
    for device in range(2):
        peer = shards[1 - device]
        cols = np.arange(256)[None, :]
        if mode == "disjoint":
            value = shards[device] + 100
            value[:, :128] = peer[:, :128]
        elif mode.startswith("empty"):
            value = peer.copy()
        elif mode in ("store", "holes", "data_store", "transpose_store"):
            mask = (cols < 32) | (cols >= 192) if mode == "holes" else cols < 192
            if mode == "data_store":
                mask = peer > 2500
            value = np.where(mask, peer + 100, peer)
        elif mode == "swap":
            value = peer.copy()
        elif mode == "vector_other":
            value = np.where(cols < 192, peer, shards[device] + 100)
        elif mode == "squeezed":
            value = peer.copy()
            value[0] = np.where(cols[0] < 192, peer[1], 7)
        else:
            limit = (
                128
                if mode == "partial_init"
                else 128 + device * 64
                if mode == "device_mask"
                else 192
            )
            mask = peer > 2500 if mode.startswith("data_") else cols < limit
            value = np.where(mask, peer, 7)
        outputs.append(value)
    return np.concatenate(outputs, axis=0)


def tail_function(mode="tail", *, mesh=None):
    shape, block = (8, 384), (8, 256)
    spec = pl.BlockSpec(block, lambda p: (0, p))

    def kernel(x, y, temp, send, recv, ack):
        peer = 1 - lax.axis_index("d")
        pl.semaphore_signal(ack, 1, device_id=(peer,))
        pl.semaphore_wait(ack, 1)
        mask = jnp.broadcast_to(pl.program_id(0) * 256 + jnp.arange(256)[None, :] < 384, block)
        value = (
            pp.load(x, (slice(None), slice(None)), mask=mask, other=0)
            if mode == "tail_mask"
            else x[...]
        )
        if mode == "tail":
            value = lax.select(mask, value, jnp.zeros(block, jnp.float32))
        temp[...] = value
        tpu.async_remote_copy(temp, y, send, recv, (peer,)).wait()

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(2,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(
            tpu.VMEM(block, jnp.float32),
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR,
        ),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    mesh = mesh or Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    return jax.shard_map(
        call, mesh=mesh, in_specs=P(None, "d"), out_specs=P(None, "d"), check_vma=False
    )


def run_cpu(case, execution):
    """Two-device CPU TPU interpreter validates the complete masked remote output."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    x = (
        np.arange(8 * 768, dtype=np.float32).reshape(8, 768)
        if case == "tail"
        else np.arange(4096, dtype=np.float32).reshape(16, 256)
    )
    params = tpu.InterpretParams(
        dma_execution_mode=execution,
        detect_races=case != "disjoint",
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        function = tail_function(case) if case == "tail" else masked_function(case)
        executable = jax.jit(function).lower(x).compile()
    print("READY", flush=True)
    result = jax.block_until_ready(executable(x))
    jax.effects_barrier()
    reference = (
        np.concatenate((x[:, 384:], x[:, :384]), axis=1) if case == "tail" else expected(x, case)
    )
    np.testing.assert_array_equal(result, reference)
    if case != "disjoint":
        assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, mode=execution, status="ok")), flush=True)
