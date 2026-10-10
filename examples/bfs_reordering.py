"""Saved AI-authored graph transformations and CPU witnesses; no online AI.

The LlamaMLP extraction is based on SGLang-JAX (Apache-2.0), commit
538abe3fdba84b586a6a2fd151cac67cc974d35f. See the provenance and configuration
in docs/development-docs/bfs-cases.md. Functions below are the saved before/after
code. Expected labels and witnesses were written before querying the comparator.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from jax_reorder import Status, compare_jaxprs


@dataclass(frozen=True)
class AICase:
    name: str
    base: Callable
    candidate: Callable
    args: tuple
    expected: Status
    reason: str
    witness: tuple  # Exact expected baseline/candidate outputs for negative cases.
    source: str = "AI-authored classic graph"


def elementwise(x, y, *, reverse=False, wrong=False):
    if reverse:
        b = lax.neg(y)
        a = lax.mul(x, x)
    else:
        a = lax.mul(x, x)
        b = lax.neg(y)
    return lax.add(a, b) if wrong else lax.mul(a, b)


def dense_bias(x, w, bias, *, reverse=False, wrong=False):
    def broadcast():
        return lax.broadcast_in_dim(bias, x.shape, (1,))

    if reverse:
        b = broadcast()
        a = lax.dot_general(x, w, (((1,), (0,)), ((), ())))
    else:
        a = lax.dot_general(x, w, (((1,), (0,)), ((), ())))
        b = broadcast()
    return lax.sub(a, b) if wrong else lax.add(a, b)


def layout_branches(x, *, reverse=False, wrong=False):
    def left():
        return lax.reshape(lax.transpose(x, (0, 1) if wrong else (1, 0)), (4,))

    def right():
        return lax.reshape(lax.slice(x, (0, 0), (1, 2)), (2,))

    if reverse:
        b, a = right(), left()
    else:
        a, b = left(), right()
    return lax.concatenate((a, b), 0)


def reduction(x, scale, *, reverse=False, wrong=False):
    if reverse:
        b = lax.broadcast_in_dim(scale, (2,), ())
        a = lax.reduce_sum(x, (1,) if wrong else (0,))
    else:
        a = lax.reduce_sum(x, (1,) if wrong else (0,))
        b = lax.broadcast_in_dim(scale, (2,), ())
    return lax.sub(a, b)


def selection(x, y, *, reverse=False, wrong=False):
    def condition():
        return lax.lt(x, jnp.zeros_like(x)) if wrong else lax.gt(x, jnp.zeros_like(x))

    if reverse:
        b, a = lax.neg(y), condition()
    else:
        a, b = condition(), lax.neg(y)
    return lax.select(a, b, x)


def result_ports(x, y, *, reverse=False, wrong=False):
    keys, values = lax.sort((x, y), num_keys=1)
    if reverse:
        b = lax.neg(keys if wrong else values)
        a = lax.add(keys, jnp.ones_like(keys))
    else:
        a = lax.add(keys, jnp.ones_like(keys))
        b = lax.neg(keys if wrong else values)
    return a, b


def diamond(x, y, *, reverse=False, wrong=False):
    shared = lax.mul(x, x if wrong else y)
    if reverse:
        b = lax.mul(y, y)
        a = lax.neg(shared)
    else:
        a = lax.neg(shared)
        b = lax.mul(y, y)
    return lax.add(shared, b), lax.mul(a, b)


def conditional(x, y, *, reverse=False, wrong=False):
    body = partial(elementwise, reverse=reverse, wrong=wrong)
    return lax.cond(x > 0, lambda z: body(*z), lambda z: lax.sub(*z), (x, y))


def linear(x, weight, mesh, last_axis):
    # LinearBase.__call__, no bias, params_dtype=float32, default out_sharding.
    target = NamedSharding(mesh, P("data", *([None] * (x.ndim - 2)), last_axis))
    return lax.dot_general(
        x,
        weight,
        (((x.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
        out_sharding=target,
    )


def llama_mlp(x, gate_weight, up_weight, down_weight, *, mesh, reverse=False, wrong=False):
    # LlamaMLP.__call__: gate/up -> up * silu(gate) -> down.
    if reverse:
        up = linear(x, up_weight, mesh, "tensor")
        gate = linear(x, gate_weight, mesh, "tensor")
    else:
        gate = linear(x, gate_weight, mesh, "tensor")
        up = linear(x, up_weight, mesh, "tensor")
    activated = jax.nn.silu(gate)
    intermediate = lax.add(up, activated) if wrong else lax.mul(up, activated)
    return linear(intermediate, down_weight, mesh, None)


def model_mesh():
    return jax.make_mesh(
        (1, 1),
        ("data", "tensor"),
        devices=jax.devices("cpu")[:1],
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def ai_cases(mesh):
    def scalar(x):
        return jnp.array(x, dtype=jnp.float32)

    matrix = scalar([[1, 2], [3, 4]])
    # Each tuple specifies a baseline and a numerically different candidate.
    specifications = [
        (
            "elementwise",
            elementwise,
            (scalar(2), scalar(3)),
            (-12.0, 1.0),
            "mul 改成 add，合流算子不同",
        ),
        (
            "dense_bias",
            dense_bias,
            (matrix, jnp.eye(2), scalar([1, 2])),
            ([[2, 4], [4, 6]], [[0, 0], [2, 2]]),
            "偏置加法改成减法",
        ),
        (
            "layout",
            layout_branches,
            (matrix,),
            ([1, 3, 2, 4, 1, 2], [1, 2, 3, 4, 1, 2]),
            "transpose 的维度参数改变",
        ),
        ("reduction", reduction, (matrix, scalar(2)), ([2, 4], [1, 5]), "归约轴改变"),
        (
            "selection",
            selection,
            (scalar([-2, 3]), scalar([5, 7])),
            ([-2, -7], [-5, 3]),
            "比较 gt 改成 lt，选择条件改变",
        ),
        (
            "result_ports",
            result_ports,
            (scalar([2, 1]), scalar([20, 10])),
            (([2, 3], [-10, -20]), ([2, 3], [-1, -2])),
            "neg 读取排序 keys 而非 values 端口",
        ),
        (
            "diamond",
            diamond,
            (scalar(2), scalar(3)),
            ((15.0, -54.0), (13.0, -36.0)),
            "共享节点 mul 的第二个输入由 y 改为 x",
        ),
        (
            "conditional",
            conditional,
            (scalar(2), scalar(3)),
            (-12.0, 1.0),
            "cond 真分支内的 mul 改成 add，分支边界保留",
        ),
    ]
    for name, base, args, witness, reason in specifications:
        yield AICase(
            name + "/reorder",
            base,
            partial(base, reverse=True),
            args,
            Status.EQUIVALENT,
            "按 a/b 或 gate/up 的角色一一对应；仅交换独立定义，操作数位置及共享关系不变",
            (),
        )
        yield AICase(
            name + "/changed",
            base,
            partial(base, wrong=True),
            args,
            Status.DIFFERENT,
            reason,
            witness,
        )
    arrays = (scalar([[1]]), scalar([[0]]), scalar([[2]]), scalar([[1]]))
    specs = (P("data", None), P(None, "tensor"), P(None, "tensor"), P("tensor", None))
    args = tuple(
        jax.device_put(a, NamedSharding(mesh, p)) for a, p in zip(arrays, specs, strict=True)
    )
    base = partial(llama_mlp, mesh=mesh)
    source = "SGLang-JAX LlamaMLP / LinearBase @ 538abe3fdba84b586a6a2fd151cac67cc974d35f"
    yield AICase(
        "llama_mlp/reorder",
        base,
        partial(base, reverse=True),
        args,
        Status.EQUIVALENT,
        "gate/up 投影独立，保留 up * silu(gate)、down 及分片精度参数",
        (),
        source,
    )
    yield AICase(
        "llama_mlp/changed",
        base,
        partial(base, wrong=True),
        args,
        Status.DIFFERENT,
        "中间 mul 改为 add；silu(0)=0，基准为 0，候选为 2",
        ([[0.0]], [[2.0]]),
        source,
    )


def leaf_arrays(value):
    return [np.asarray(x) for x in jax.tree.leaves(value)]


def verify_case(case):
    before = jax.make_jaxpr(case.base)(*case.args)
    after = jax.make_jaxpr(case.candidate)(*case.args)
    # Replay numerical evidence independently before consulting the comparator.
    a, b = case.base(*case.args), case.candidate(*case.args)
    if case.witness:
        for actual, expected in zip((a, b), case.witness, strict=True):
            actual_leaves = leaf_arrays(actual)
            expected_leaves = list(expected) if isinstance(actual, tuple) else [expected]
            for x, y in zip(actual_leaves, expected_leaves, strict=True):
                np.testing.assert_array_equal(x, y)
        assert any(
            not np.array_equal(x, y) for x, y in zip(leaf_arrays(a), leaf_arrays(b), strict=True)
        )
    else:
        for x, y in zip(leaf_arrays(a), leaf_arrays(b), strict=True):
            np.testing.assert_array_equal(x, y)
    comparison = compare_jaxprs(before, after)
    assert comparison.status == case.expected, (case.name, comparison)
    return comparison


def main():
    mesh = model_mesh()
    print(f"JAX {jax.__version__}; backend={jax.default_backend()}; x64={jax.config.x64_enabled}")
    with jax.set_mesh(mesh):
        for case in ai_cases(mesh):
            result = verify_case(case)
            print(f"{case.name}: AI={case.expected.value}, program={result.status.value}")
            print(f"  {case.reason}")
            if case.witness:
                print(f"  CPU witness: {case.witness[0]} -> {case.witness[1]}")


if __name__ == "__main__":
    main()
