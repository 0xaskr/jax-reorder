"""Static protocol regressions with independently specified array outputs."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as tpu


def regular_values(x, y, sem):
    pl.semaphore_signal(sem, 3)
    first = pl.semaphore_read(sem)
    pl.semaphore_wait(sem, 1)
    pl.semaphore_wait(sem, 2)
    pl.semaphore_signal(sem, 2)
    pl.semaphore_wait(sem, 2)
    y[:] = x[:] + first + pl.semaphore_read(sem)


def regular_control(x, y, sem):
    pl.semaphore_signal(sem, 3)
    pl.semaphore_wait(sem, 1)
    pl.semaphore_wait(sem, 2)
    pl.semaphore_signal(sem, 2)
    pl.semaphore_wait(sem, 2)
    y[:] = x[:] + 3


def local_regular_function(kernel=regular_values, *, interpret=False):
    spec = pl.BlockSpec((128,), lambda p: (p,))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.REGULAR,),
        interpret=interpret,
    )


def acknowledged_exchange(*, early=False):
    def kernel(x, y, temporary, send, receive, ack):
        target = 1 - lax.axis_index("d")
        first = tpu.async_remote_copy(x.at[:128], temporary.at[:128], send, receive, (target,))
        first.wait_send()
        second = tpu.make_async_remote_copy(
            x.at[128:], temporary.at[:128], send, receive, (target,)
        )
        if early:
            second.start()
        first.wait_recv()
        y[:128] = temporary[:128]
        pl.semaphore_signal(ack, 1, device_id=(target,))
        pl.semaphore_wait(ack, 1)
        if not early:
            second.start()
        second.wait()
        y[128:] = temporary[:128]

    return kernel


def mixed_dma(*, early=False):
    def kernel(x, y, temporary, local, send, receive):
        copy = tpu.async_copy(x, temporary, local)
        if not early:
            copy.wait()
        tpu.async_remote_copy(temporary, y, send, receive, (1 - lax.axis_index("d"),)).wait()
        if early:
            copy.wait()

    return kernel


def shared_local_remote_counter(x, y, temporary, send, receive):
    # Independent destinations, identical logical count units, one full-count wait.
    tpu.async_copy(x.at[:128], temporary.at[:128], receive)
    copy = tpu.async_remote_copy(
        x.at[128:], temporary.at[128:], send, receive, (1 - lax.axis_index("d"),)
    )
    copy.wait_send()
    tpu.make_async_copy(x, temporary, receive).wait()
    y[:] = temporary[:]


def signaled_completion(*, early=False):
    def kernel(x, y, temporary, send, receive, ready):
        target = 1 - lax.axis_index("d")
        copy = tpu.async_remote_copy(x, temporary, send, receive, (target,))
        if not early:
            copy.wait_recv()
        # A local signal carries no completion guarantee for an outstanding DMA.
        pl.semaphore_signal(ready, 1)
        pl.semaphore_wait(ready, 1)
        y[:] = temporary[:]
        if early:
            copy.wait_recv()
        copy.wait_send()

    return kernel


def regular_remote(x, y, sem):
    target = 1 - lax.axis_index("d")
    pl.semaphore_signal(sem, 2, device_id=target)
    pl.semaphore_wait(sem, 1)
    pl.semaphore_wait(sem, 1)
    y[:] = x[:] + pl.semaphore_read(sem)


def regular_remote_control(x, y, sem):
    target = 1 - lax.axis_index("d")
    pl.semaphore_signal(sem, 2, device_id=target)
    pl.semaphore_wait(sem, 1)
    pl.semaphore_wait(sem, 1)
    y[:] = x[:]
