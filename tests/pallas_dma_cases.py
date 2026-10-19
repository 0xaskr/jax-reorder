"""Native source cases shared with the independent interpreter subprocess."""

from jax.experimental.pallas import tpu as pltpu

from tests.test_pallas_dma import copy, function


def split_kernel(*, size=256, chunks=2, order=None, waits="total", swapped=False):
    width = size // chunks
    order = tuple(range(chunks)) if order is None else order

    def kernel(x, y, semaphore):
        transfers = []
        for i in order:
            src = (chunks - i - 1 if swapped else i) * width
            dst = i * width
            transfer = pltpu.async_copy(x.at[src : src + width], y.at[dst : dst + width], semaphore)
            transfers.append(transfer)
            if waits == "serial":
                transfer.wait()
        if waits == "total":
            pltpu.make_async_copy(x, y, semaphore).wait()
        elif waits == "parts":
            for transfer in transfers:
                transfer.wait()

    return kernel


def relayed(x, y, temporary, semaphore):
    pltpu.async_copy(x.at[:128], temporary.at[:128], semaphore)
    pltpu.async_copy(x.at[128:], temporary.at[128:], semaphore)
    pltpu.make_async_copy(x, temporary, semaphore).wait()
    pltpu.async_copy(temporary, y, semaphore).wait()


def disjoint_reuse(x, y, temporary, first, second):
    temporary[:] = x[:]
    a = pltpu.async_copy(temporary.at[:128], y.at[:128], first)
    b = pltpu.async_copy(x.at[128:], temporary.at[128:], second)
    b.wait()
    a.wait()
    pltpu.async_copy(temporary.at[128:], y.at[128:], first).wait()


def sample_case(name, interpret=False):
    options = {"size": 256, "block": 256, "interpret": interpret}
    if name == "relay":
        return function(relayed, arrays=1, **options)
    if name == "disjoint_reuse":
        return function(disjoint_reuse, arrays=1, semaphores=2, **options)
    if name == "whole":
        return function(copy, **options)
    return function(
        split_kernel(waits="parts" if name == "parts" else "total", swapped=name == "swapped"),
        **options,
    )
