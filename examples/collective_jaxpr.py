"""在多个逻辑 CPU 设备上验证带通信的 jaxpr。

运行：uv run python examples/collective_jaxpr.py --show-jaxpr
默认创建两个 CPU 设备，也可用 --devices 4 检查四设备通信。
模块导入时不初始化后端，pytest 可以复用用例而不改变自己的设备配置。
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
from jax import lax
from jax._src import core
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import Status, canonicalize, compare_jaxprs

AXIS = "workers"


def independent(x, y):
    """两条通信相互独立，但最终减法的左右操作数固定。"""
    total = lax.psum(x, AXIS)
    maximum = lax.pmax(y, AXIS)
    return lax.sub(total, maximum)


def reordered(x, y):
    """先求最大值再求和，只改变独立通信的定义顺序。"""
    maximum = lax.pmax(y, AXIS)
    total = lax.psum(x, AXIS)
    return lax.sub(total, maximum)


def map_body(body, mesh, *, check_vma=True):
    """保留整个 shard_map 边界；输出沿首维拼接各设备的结果。"""
    return jax.shard_map(
        body, mesh=mesh, in_specs=P(AXIS, None), out_specs=P(AXIS, None), check_vma=check_vma
    )


@dataclass(frozen=True)
class Case:
    """图标签与数值预期分开；primitive 集合用于确认通信确实进入追踪图。"""

    name: str
    original: Callable
    candidate: Callable
    expected: Status
    same_samples: bool
    primitives: frozenset[str]


def cases(mesh, *, check_vma=True):
    """组合显式通信、本地计算与嵌套区域，保存 AI 给出的预期判断。"""
    count = mesh.size
    ring = [(i, (i + 1) % count) for i in range(count)]
    identity = [(i, i) for i in range(count)]
    result = []

    def add(name, left, right, expected, same, *primitives, vma=check_vma):
        result.append(
            Case(
                name,
                map_body(left, mesh, check_vma=vma),
                map_body(right, mesh, check_vma=vma),
                expected,
                same,
                frozenset(primitives),
            )
        )

    # 开启变化类型检查时，psum 会追踪为 psum_invariant；二者不能按名字合并。
    sum_name = "psum_invariant" if check_vma else "psum"
    add("独立的两种通信换序", independent, reordered, Status.EQUIVALENT, True, sum_name, "pmax")

    def sum_pair(x, y):
        a = lax.psum(x, AXIS)
        b = lax.psum(y, AXIS)
        return lax.sub(a, b)

    def sum_pair_reordered(x, y):
        b = lax.psum(y, AXIS)
        a = lax.psum(x, AXIS)
        return lax.sub(a, b)

    add("相同通信原语换序", sum_pair, sum_pair_reordered, Status.EQUIVALENT, True, sum_name)

    def with_local(communication, reverse=False):
        """固定依赖和加法操作数，只切换通信分支与本地分支的定义顺序。"""

        def body(x, y):
            if reverse:
                local = lax.tanh(y)
                value = communication(x)
            else:
                value = communication(x)
                local = lax.tanh(y)
            return lax.add(value, local)

        return body

    def scatter_gather(x):
        """先归约并分散，再收集；此链保留真实数据依赖。"""
        part = lax.psum_scatter(x, AXIS, scatter_dimension=0, tiled=True)
        return lax.all_gather(part, AXIS, axis=0, tiled=True)

    for name, communication, primitive_names in (
        ("归约与本地计算换序", lambda x: lax.psum(x, AXIS), (sum_name,)),
        ("通信链与本地计算换序", scatter_gather, ("reduce_scatter", "all_gather")),
        ("置换与本地计算换序", lambda x: lax.ppermute(x, AXIS, ring), ("ppermute",)),
        (
            "all_to_all 与本地计算换序",
            lambda x: lax.all_to_all(x, AXIS, 0, 0, tiled=True),
            ("all_to_all",),
        ),
        ("最小值归约与本地计算换序", lambda x: lax.pmin(x, AXIS), ("pmin",)),
    ):
        add(
            name,
            with_local(communication),
            with_local(communication, True),
            Status.EQUIVALENT,
            True,
            *primitive_names,
        )

    add(
        "psum 改为 pmax",
        lambda x, y: lax.psum(x, AXIS),
        lambda x, y: lax.pmax(x, AXIS),
        Status.DIFFERENT,
        False,
        sum_name,
        "pmax",
    )
    add(
        "通信输入绑定改变",
        sum_pair,
        lambda x, y: sum_pair(y, x),
        Status.DIFFERENT,
        False,
        sum_name,
    )
    add(
        "置换路径改变",
        lambda x, y: lax.ppermute(x, AXIS, ring),
        lambda x, y: lax.ppermute(x, AXIS, identity),
        Status.DIFFERENT,
        False,
        "ppermute",
    )
    # JAX 的设备子组要求关闭 check_vma；该条件是追踪环境的一部分，两边保持一致。
    add(
        "归约设备分组改变",
        lambda x, y: lax.psum(x, AXIS),
        lambda x, y: lax.psum(x, AXIS, axis_index_groups=[[i] for i in range(count)]),
        Status.DIFFERENT,
        False,
        "psum",
        vma=False,
    )
    add(
        "all_to_all 维度改变",
        lambda x, y: lax.all_to_all(x, AXIS, 0, 0, tiled=True),
        lambda x, y: lax.all_to_all(x, AXIS, 1, 1, tiled=True),
        Status.DIFFERENT,
        False,
        "all_to_all",
    )
    add(
        "加法操作数交换",
        with_local(lambda x: lax.psum(x, AXIS)),
        lambda x, y: lax.add(lax.tanh(y), lax.psum(x, AXIS)),
        Status.DIFFERENT,
        True,
        sum_name,
    )
    add(
        "all-reduce 改为 scatter 加 gather",
        lambda x, y: lax.psum(x, AXIS),
        lambda x, y: scatter_gather(x),
        Status.DIFFERENT,
        True,
        sum_name,
        "reduce_scatter",
        "all_gather",
    )
    add(
        "激活跨越归约",
        lambda x, y: lax.psum(lax.tanh(x), AXIS),
        lambda x, y: lax.tanh(lax.psum(x, AXIS)),
        Status.DIFFERENT,
        False,
        sum_name,
    )

    def unused(x, y):
        """未使用通信不应被规范化器静默删除；本轮返回 unknown。"""
        lax.psum(y, AXIS)
        return x

    add("未使用的通信", lambda x, y: x, unused, Status.UNKNOWN, True, sum_name)

    def scan(body):
        """所有设备固定执行三轮；通信保持在 scan 区域内。"""
        # 本地 y 保持 carry 的 varying 类型，与初始分片 x 一致。
        return lambda x, y: lax.scan(
            lambda c, _: (lax.add(body(c, y), y), None), x, None, length=3
        )[0]

    def cond(body):
        """用全局归约构造一致谓词，避免不同设备进入不同的通信分支。"""

        def run(x, y):
            flag = lax.psum(lax.reduce_sum(x, (0, 1)), AXIS) > 0
            # 显式传参固定源码接口；下方闭包版本另验证自动捕获参数的同步换序。
            return lax.cond(
                flag,
                lambda args: lax.add(body(*args), args[1]),
                lambda args: lax.sub(*args),
                (x, y),
            )

        return run

    for name, wrap in (("jit", jax.jit), ("scan", scan), ("cond", cond)):
        add(
            f"{name} 内通信换序",
            wrap(independent),
            wrap(reordered),
            Status.EQUIVALENT,
            True,
            sum_name,
            "pmax",
        )

    def captured_cond(body):
        """正例：捕获位置改变，但调用实参和分支形参的绑定关系保持不变。"""

        def run(x, y):
            flag = lax.psum(lax.reduce_sum(x, (0, 1)), AXIS) > 0
            return lax.cond(flag, lambda: lax.add(body(x, y), y), lambda: lax.sub(x, y))

        return run

    add(
        "cond 闭包捕获顺序改变",
        captured_cond(independent),
        captured_cond(reordered),
        Status.EQUIVALENT,
        True,
        sum_name,
        "pmax",
    )
    return tuple(result)


def parallel_matmul(mesh, reverse=False, missing_sum=False):
    """按收缩维分片的矩阵乘法；各设备的局部乘积经 psum 合并后加偏置。"""

    def body(x, weight, bias):
        if reverse:
            expanded = lax.broadcast_in_dim(bias, (x.shape[0], weight.shape[1]), (1,))
        partial = lax.dot_general(x, weight, (((1,), (0,)), ((), ())))
        total = partial if missing_sum else lax.psum(partial, AXIS)
        if not reverse:
            expanded = lax.broadcast_in_dim(bias, total.shape, (1,))
        return lax.add(total, expanded)

    # 输出拼接所有设备的局部结果，缺失 psum 时仍有合法输出签名，便于提供数值反例。
    return jax.shard_map(
        body, mesh=mesh, in_specs=(P(None, AXIS), P(AXIS, None), P()), out_specs=P(AXIS, None)
    )


def equations(graph):
    """遍历参数容器里的嵌套图，用于打印与检查通信原语是否真实存在。"""
    for equation in graph.eqns:
        yield equation
        pending = list(equation.params.values())
        while pending:
            value = pending.pop()
            if isinstance(value, core.Jaxpr):
                yield from equations(value)
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
            elif isinstance(value, dict):
                pending.extend(value.values())


def check_values(left, right):
    """核对每个输出的 shape、dtype 和数值；不把采样当作等价性证明。"""
    a, b = jax.tree.leaves(left), jax.tree.leaves(right)
    assert len(a) == len(b), "输出叶子数改变"
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype, "输出签名改变"
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)


def run_case(case, samples, *, show_jaxpr=False):
    """检查图标签和幂等性，分别编译两侧原函数及重排图，并运行全部样本。"""
    graphs = [jax.make_jaxpr(fn)(*samples[0]) for fn in (case.original, case.candidate)]
    found = {eq.primitive.name for graph in graphs for eq in equations(graph)}
    assert case.primitives <= found, f"通信没有进入图：{case.primitives - found}"
    comparison = compare_jaxprs(*graphs)
    assert comparison.status == case.expected, comparison
    normalized = [canonicalize(graph) for graph in graphs]
    for item in normalized:
        core.check_jaxpr(item.jaxpr)
        again = canonicalize(item.jaxpr)
        assert again.graph == item.graph, "规范化不幂等"
        # JAX 子图对象自身不按结构比较；完整规范结构负责验证嵌套区域。
        assert again.equation_order == tuple(range(len(item.jaxpr.eqns))), "方程再次换序"

    original_functions = [jax.jit(case.original), jax.jit(case.candidate)]
    normalized_functions = [
        jax.jit(lambda *args, graph=item.jaxpr: core.eval_jaxpr(graph, graph.consts, *args))
        for item in normalized
    ]
    witness = None
    for inputs in samples:
        a, b = [fn(*inputs) for fn in original_functions]
        assert jax.tree.structure(a) == jax.tree.structure(b), "输出 pytree 改变"
        for value, fn in zip((a, b), normalized_functions, strict=True):
            check_values(value, fn(*jax.tree.leaves(inputs)))
        if case.same_samples:
            check_values(a, b)
        elif witness is None and any(
            not np.allclose(x, y, rtol=1e-5, atol=1e-5)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
        ):
            witness = (inputs, a, b)
    assert case.same_samples or witness is not None, "未找到预期数值反例"

    print(f"PASS {case.name}: {comparison.status.value}；{len(samples)} 组样本重排执行通过")
    if comparison.difference:
        print(f"  图差异：{comparison.difference}")
    for issue in comparison.issues:
        print(f"  无法判定：{issue.path}: {issue.reason}")
    if witness is not None:
        inputs, a, b = witness
        print(f"  反例输入：{[np.asarray(x).tolist() for x in inputs]}")
        print(f"  基准输出：{np.asarray(a).tolist()}；候选输出：{np.asarray(b).tolist()}")
    if show_jaxpr:
        for label, graph, item in zip(("基准", "候选"), graphs, normalized, strict=True):
            print(f"  {label}原图：\n{graph}\n  {label}重排后：\n{item.jaxpr}")
    return comparison


def make_samples(count, trials, seed, dtype=np.float32):
    """每设备有 count×count 个元素，使两个轴均可用于 all_to_all 和 scatter。"""
    rng = np.random.default_rng(seed)
    # 首组均为正数，后续交替正负，让 cond 的两条分支均被执行。
    return [
        tuple(
            (rng.integers(1, 8, size=(count * count, count)) * (-1 if i % 2 else 1)).astype(dtype)
            for _ in range(2)
        )
        for i in range(trials)
    ]


def main():
    """独立进程在首次使用后端前设置设备数量；任一断言失败即返回非零退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-jaxpr", action="store_true")
    args = parser.parse_args()
    if args.devices < 2 or args.trials < 1:
        parser.error("--devices 至少为 2，--trials 至少为 1")
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_num_cpu_devices", args.devices)
    devices = jax.devices("cpu")
    assert len(devices) == args.devices, "后端设备配置没有生效"
    mesh = Mesh(np.array(devices), (AXIS,))
    print(f"JAX {jax.__version__}，{len(devices)} 个逻辑 CPU 设备")
    samples = make_samples(mesh.size, args.trials, args.seed)
    all_cases = cases(mesh)
    for case in all_cases:
        run_case(case, samples, show_jaxpr=args.show_jaxpr)

    rng = np.random.default_rng(args.seed)
    layer_samples = [
        tuple(
            rng.normal(size=shape).astype(np.float32)
            for shape in ((2, mesh.size * 2), (mesh.size * 2, 3), (3,))
        )
        for _ in range(args.trials)
    ]
    for label, candidate, expected in (
        ("分片矩阵乘法的偏置广播换序", parallel_matmul(mesh, reverse=True), Status.EQUIVALENT),
        ("分片矩阵乘法遗漏 all-reduce", parallel_matmul(mesh, missing_sum=True), Status.DIFFERENT),
    ):
        case = Case(
            label,
            parallel_matmul(mesh),
            candidate,
            expected,
            expected == Status.EQUIVALENT,
            frozenset({"dot_general", "psum_invariant"}),
        )
        run_case(case, layer_samples, show_jaxpr=args.show_jaxpr)
    print(f"全部通过：{len(all_cases) + 2} 个通信组合；数值采样不替代图比较。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
