"""One-call remote kernels with explicit send/receive phases on a fixed mesh."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def exchange(x, y, send, receive):
    target = lax.sub(jnp.int32(1), lax.axis_index("d"))
    pltpu.async_remote_copy(x, y, send, receive, (target,)).wait()


def request_reply(*, delayed=False):
    def kernel(x, y, request_send, request_receive, reply_send, reply_receive):
        device = lax.axis_index("d")
        request = pltpu.make_async_remote_copy(x, y, request_send, request_receive, (jnp.int32(1),))
        reply = pltpu.make_async_remote_copy(y, y, reply_send, reply_receive, (jnp.int32(0),))

        def a():
            if delayed:
                reply.wait_recv()
                request.start()
            else:
                request.start()
                reply.wait_recv()
            request.wait_send()

        def b():
            request.wait_recv()
            reply.start()
            reply.wait_send()

        lax.cond(device == 0, a, b)

    return kernel


def three_device_route(*, relay):
    def kernel(x, y, first_send, first_receive, second_send, second_receive):
        first = pltpu.make_async_remote_copy(x, y, first_send, first_receive, (jnp.int32(1),))
        second = pltpu.make_async_remote_copy(
            y if relay else x, y, second_send, second_receive, (jnp.int32(2),)
        )

        def a():
            y[:] = x[:]
            first.start()
            if not relay:
                second.start()
                second.wait_send()
            first.wait_send()

        def b():
            first.wait_recv()
            if relay:
                second.start()
                second.wait_send()

        lax.switch(lax.axis_index("d"), (a, b, second.wait_recv))

    return kernel


def remote_function(kernel, *, semaphores=2, interpret=False, compiler=None, devices=2):
    mesh = Mesh(np.array(jax.devices("cpu")[:devices]), ("d",))
    assert mesh.size == devices
    spec = pl.BlockSpec((128,), lambda p: (p,))
    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(pltpu.SemaphoreType.DMA,) * semaphores,
        compiler_params=compiler,
        interpret=interpret,
    )
    return jax.shard_map(function, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)


def remote_graph(kernel=exchange, **options):
    return jax.make_jaxpr(remote_function(kernel, **options))(
        jax.ShapeDtypeStruct((128 * options.get("devices", 2),), jnp.float32)
    )
