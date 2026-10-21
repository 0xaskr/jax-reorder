"""One-call remote kernels with explicit send/receive phases on a fixed mesh."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder.core.index_domain import IndexExpr
from jax_reorder.core.memory import Predicate, ValueExpr

PARTITIONED = P("d")


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


def remote_function(
    kernel,
    *,
    semaphores=2,
    regular=0,
    interpret=False,
    compiler=None,
    devices=2,
    size=128,
    arrays=0,
    inputs=1,
    outputs=1,
    in_spec=PARTITIONED,
    out_spec=PARTITIONED,
):
    mesh = Mesh(np.array(jax.devices("cpu")[:devices]), ("d",))
    assert mesh.size == devices
    spec = pl.BlockSpec((size,), lambda p: (p,))
    shape = jax.ShapeDtypeStruct((size,), jnp.float32)
    function = pl.pallas_call(
        kernel,
        out_shape=shape if outputs == 1 else (shape,) * outputs,
        grid=(1,),
        in_specs=(spec,) * inputs,
        out_specs=spec if outputs == 1 else (spec,) * outputs,
        scratch_shapes=(pltpu.VMEM((size,), jnp.float32),) * arrays
        + (pltpu.SemaphoreType.DMA,) * semaphores
        + (pltpu.SemaphoreType.REGULAR,) * regular,
        compiler_params=compiler,
        interpret=interpret,
    )
    return jax.shard_map(function, mesh=mesh, in_specs=in_spec, out_specs=out_spec, check_vma=False)


def remote_graph(kernel=exchange, **options):
    size = options.get("size", 128)
    if options.get("in_spec", P("d")) == P("d"):
        size *= options.get("devices", 2)
    return jax.make_jaxpr(remote_function(kernel, **options))(
        *(jax.ShapeDtypeStruct((size,), jnp.float32),) * options.get("inputs", 1)
    )


def exchange_waits(*, receive_first=False, self_target=False):
    def kernel(x, y, send, receive):
        device = lax.axis_index("d")
        target = device if self_target else 1 - device
        transfer = pltpu.async_remote_copy(x, y, send, receive, (target,))
        if receive_first:
            transfer.wait_recv()
            transfer.wait_send()
        else:
            transfer.wait()

    return kernel


def reuse_source(*, early=False, restore=False):
    def kernel(x, y, temporary, s0, r0, s1, r1):
        target = 1 - lax.axis_index("d")
        temporary[:] = x[:]
        first = pltpu.async_remote_copy(temporary.at[:128], y.at[:128], s0, r0, (target,))
        if not early:
            first.wait_send()
        temporary[:] = x[:] + 1
        if restore:
            temporary[:] = x[:]
        if early:
            first.wait_send()
        second = pltpu.async_remote_copy(temporary.at[128:], y.at[128:], s1, r1, (target,))
        first.wait_recv()
        second.wait()

    return kernel


def sliced_exchange(*, swapped=True, split_wait=False):
    def kernel(x, y, send, receive):
        target = 1 - lax.axis_index("d")
        a, b = (y.at[128:], y.at[:128]) if swapped else (y.at[:128], y.at[128:])
        first = pltpu.async_remote_copy(x.at[:128], a, send, receive, (target,))
        second = pltpu.async_remote_copy(x.at[128:], b, send, receive, (target,))
        if split_wait:
            first.wait()
            second.wait()
        else:
            pltpu.make_async_remote_copy(x, y, send, receive, (target,)).wait()

    return kernel


def partial_exchange(x, y, temporary, send, receive):
    device = lax.axis_index("d")
    pltpu.async_remote_copy(x, temporary, send, receive, (1 - device,)).wait()
    y[:128] = temporary[128:] + jnp.arange(128, dtype=jnp.int32).astype(jnp.float32)
    y[128:] = temporary[:128] + jnp.float32(device)


def projection_array(expression, inputs, size):
    """Small-array oracle only: independently evaluate the emitted value DAG."""
    _coordinate = 0
    done = {}

    def evaluate(value):
        if id(value) in done:
            return done[id(value)]
        if type(value) is bool:
            result = value
        elif isinstance(value, IndexExpr):
            children = [evaluate(c) for c in value.args]
            if value.kind == "constant":
                result = int(value.value)
            elif value.kind == "axis":
                assert value.value == "global"
                result = _coordinate
            elif value.kind == "sum":
                result = sum(children)
            elif value.kind == "scale":
                result = children[0] * int(value.value)
            elif value.kind == "div":
                result = children[0] // int(value.value)
            else:
                assert value.kind == "mod"
                result = children[0] % int(value.value)
        elif isinstance(value, Predicate):
            assert value.operation == "lt"
            result = evaluate(value.operands[0]) < evaluate(value.operands[1])
        else:
            assert isinstance(value, ValueExpr)
            if value.operation == "select_n":
                result = evaluate(value.operands[int(evaluate(value.operands[0])) + 1])
                done[id(value)] = result
                return result
            children = [evaluate(c) for c in value.operands]
            if value.operation == "input":
                if children:
                    assert 0 <= children[0] < inputs[value.input].size
                    result = inputs[value.input][children[0]]
                else:
                    result = inputs[value.input]
            elif value.operation == "constant_array":
                data = np.frombuffer(value.data, dtype=value.type.dtype)
                result = data[children[0]]
            elif value.operation == "literal":
                result = np.frombuffer(value.data, dtype=value.type.dtype)[0]
            elif value.operation == "convert_element_type":
                result = np.asarray(children[0]).astype(value.type.dtype)
            else:
                result = {
                    "add": np.add,
                    "sub": np.subtract,
                    "mul": np.multiply,
                    "gt": np.greater,
                    "lt": np.less,
                    "eq": np.equal,
                    "clamp": lambda lo, v, hi: np.clip(v, lo, hi),
                }[value.operation](*children)
        done[id(value)] = result
        return result

    result = []
    for _coordinate in range(size):
        done.clear()
        result.append(evaluate(expression))
    return np.asarray(result)
