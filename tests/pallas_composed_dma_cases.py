"""Replayable source cases for padded Ref and exact DMA footprint composition."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def local_holes(*, mode="normal", destination_holes=False, interpret=False):
    shape = (32, 128)

    def kernel(x, y, scratch, sem):
        if mode in ("partial_source", "missing_source"):
            scratch[:8, :] = x[:8, :]
            if mode == "partial_source":
                scratch[16:24, :] = x[16:24, :]
        else:
            scratch[...] = x[...]
        if mode != "missing_destination":
            y[...] = x[...] + 10000
        source = scratch.reshape((2, 16, 128)).at[:, :8, :]
        target = (
            y.reshape((2, 16, 128)).at[:, :8, :]
            if destination_holes
            else y.at[:16, :].reshape((2, 8, 128))
        )
        copy = tpu.async_copy(source, target, sem)
        if mode == "source_hole":
            scratch[8:16, :] = x[8:16, :] + 1
        elif mode == "destination_hole":
            y[8:16, :] = x[8:16, :] + 1
        elif mode == "source_overlap":
            scratch[:8, :] = x[:8, :]
        elif mode == "destination_overlap":
            y[:8, :] = x[:8, :]
        if mode != "missing_wait":
            copy.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        scratch_shapes=(tpu.VMEM(shape, jnp.float32), tpu.SemaphoreType.DMA),
        interpret=interpret,
    )


def local_expected(x, *, destination_holes=False, mode="normal"):
    result = x + 10000
    selected = x.reshape(4, 8, 128)[::2].reshape(16, 128)
    if destination_holes:
        result.reshape(4, 8, 128)[::2] = selected.reshape(2, 8, 128)
    else:
        result[:16] = selected
    if mode == "destination_hole":
        result[8:16] = x[8:16] + 1
    return result


def padded_view(*, mode="reshape", limit=5, interpret=False):
    def kernel(x, y):
        view = x.reshape((2, 2, 4))
        if mode == "transpose":
            raw = view.transpose((0, 2, 1))[...]
            positions = jnp.arange(8).reshape(2, 4).T
        elif mode == "slice":
            raw = view.at[:, 1, :].reshape((2, 4))[...]
            positions = jnp.arange(4) + 4
        else:
            raw = view[...]
            positions = jnp.arange(8).reshape(2, 4)
        clean = jnp.where(positions < limit, raw, 0)
        y[...] = jnp.sum(clean, axis=tuple(range(1, clean.ndim))).reshape((2, 1))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((2, 1), jnp.float32),
        in_specs=(pl.BlockSpec((2, 8), lambda: (0, 0)),),
        out_specs=pl.BlockSpec((2, 1), lambda: (0, 0)),
        interpret=interpret,
    )


def remote_kernel(*, mode="reshape", wrong=False):
    def kernel(x, y, scratch, local, send, receive):
        peer = jax.lax.axis_index("d") if wrong else 1 - jax.lax.axis_index("d")
        if mode == "indexed":
            tpu.async_remote_copy(x.at[:].at[128:], y.at[:].at[:128], send, receive, (peer,)).wait()
            tpu.async_remote_copy(x.at[:].at[:128], y.at[:].at[128:], send, receive, (peer,)).wait()
        elif mode == "mixed":
            tpu.async_copy(
                x.reshape((2, 128)).transpose((1, 0)), scratch.reshape((128, 2)), local
            ).wait()
            tpu.async_remote_copy(scratch, y, send, receive, (peer,)).wait()
        elif mode in ("holes", "hole_write", "overlap", "missing_wait"):
            source = x.reshape((2, 2, 128)).at[:, 0, :].reshape((256,))
            target = y.reshape((2, 2, 128)).at[:, 0, :].reshape((256,))
            transfer = tpu.async_remote_copy(source, target, send, receive, (peer,))
            y[128:256] = x[128:256]
            y[384:512] = x[384:512]
            if mode == "overlap":
                y[:128] = x[:128]
            if mode != "missing_wait":
                transfer.wait()
        else:
            source = x.reshape((2, 128))
            if mode == "transpose":
                source = source.transpose((1, 0))
                target = y.reshape((128, 2))
            else:
                target = y.reshape((2, 128))
            tpu.async_remote_copy(source, target, send, receive, (peer,)).wait()

    return kernel


def remote_expected(x, *, mode="reshape", wrong=False):
    local = x.reshape(2, -1)
    source = local if wrong else local[::-1]
    if mode == "indexed":
        return source.reshape(2, 2, 128)[:, ::-1].reshape(-1)
    if mode in ("transpose", "mixed"):
        return source.reshape(2, 2, 128).transpose(0, 2, 1).reshape(-1)
    if mode in ("holes", "hole_write"):
        result = local.copy().reshape(2, 4, 128)
        result[:, ::2] = source.reshape(2, 4, 128)[:, ::2]
        return result.reshape(-1)
    return np.asarray(source).reshape(-1)


def remote_case_function(*, mode="reshape", mesh=None):
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    if mesh is None:
        mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",))
    size = 512 if mode in ("holes", "overlap", "missing_wait") else 256
    spec = pl.BlockSpec((size,), lambda p: (p,))
    function = pl.pallas_call(
        remote_kernel(mode=mode),
        out_shape=jax.ShapeDtypeStruct((size,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.VMEM((size,), jnp.float32),) + (tpu.SemaphoreType.DMA,) * 3,
    )
    return jax.shard_map(function, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)


def native_padded_view(*, limit=384, interpret=False):
    def kernel(x, y):
        raw = x.reshape((2, 8, 512))[...]
        clean = jnp.where(jnp.arange(512) < limit, raw, 0)
        y[...] = jnp.broadcast_to(jnp.sum(clean, axis=2).reshape((16, 1)), (16, 128))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
        grid=(1,),
        in_specs=(pl.BlockSpec((16, 512), lambda p: (0, 0)),),
        out_specs=pl.BlockSpec((16, 128), lambda p: (0, 0)),
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
        interpret=interpret,
    )


def run_cpu(case, mode):
    """Official TPU CPU interpreter oracle, isolated from native/backend probes."""
    import json
    from functools import partial

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    if case.startswith("local"):
        holes = case == "local_holes"
        x = np.arange(4096, dtype=np.float32).reshape(32, 128)
        build = partial(
            local_holes,
            destination_holes=holes,
            mode="destination_hole" if holes else "source_hole",
        )
        expected = local_expected(
            x, destination_holes=holes, mode="destination_hole" if holes else "source_hole"
        )
    elif case == "padding":
        x = np.arange(16 * 384, dtype=np.float32).reshape(16, 384)
        build = native_padded_view
        expected = np.broadcast_to(x.sum(axis=1, keepdims=True), (16, 128))
    else:
        size = 512 if case == "holes" else 256
        x = np.arange(2 * size, dtype=np.float32)
        build = partial(remote_case_function, mode=case)
        expected = remote_expected(x, mode=case)
    with tpu.force_tpu_interpret_mode(params):
        executable = jax.jit(build()).lower(x).compile()
    print("READY", flush=True)
    output = jax.block_until_ready(executable(x))
    jax.effects_barrier()
    np.testing.assert_array_equal(output, expected)
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, mode=mode, status="ok")), flush=True)
