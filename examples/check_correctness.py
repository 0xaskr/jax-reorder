"""手动验证图比较与重排：uv run python examples/check_correctness.py。

可修改 original、各候选函数和 make_inputs 来测试自己的计算。
数值采样是辅助检查；图不等价也可能在所有这些样本上数值相同。
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import core

from jax_reorder import Status, canonicalize, compare_jaxprs


def original(x, weight, bias):
    """基准计算：点积 → 激活 → 加偏置 → 按行归约；偏置广播是独立分支。"""
    projected = lax.dot_general(x, weight, (((1,), (0,)), ((), ())))
    expanded = lax.broadcast_in_dim(bias, (x.shape[0], weight.shape[1]), (1,))
    activated = lax.tanh(projected)
    return lax.reduce_sum(lax.add(activated, expanded), (1,))


def reordered(x, weight, bias):
    # 正例：先算独立的 bias 广播，操作数及依赖关系保持不变。
    expanded = lax.broadcast_in_dim(bias, (x.shape[0], weight.shape[1]), (1,))
    projected = lax.dot_general(x, weight, (((1,), (0,)), ((), ())))
    activated = lax.tanh(projected)
    return lax.reduce_sum(lax.add(activated, expanded), (1,))


def changed(x, weight, bias):
    # 负例：偏置加法改为减法，图和数值结果都不同。
    projected = lax.dot_general(x, weight, (((1,), (0,)), ((), ())))
    expanded = lax.broadcast_in_dim(bias, (x.shape[0], weight.shape[1]), (1,))
    activated = lax.tanh(projected)
    return lax.reduce_sum(lax.sub(activated, expanded), (1,))


def swapped_operands(x, weight, bias):
    # 本轮不使用交换律：样本数值相同，图仍应判断为 different。
    projected = lax.dot_general(x, weight, (((1,), (0,)), ((), ())))
    expanded = lax.broadcast_in_dim(bias, (x.shape[0], weight.shape[1]), (1,))
    activated = lax.tanh(projected)
    return lax.reduce_sum(lax.add(expanded, activated), (1,))


def unused_branch(x, weight, bias):
    # 完整图保留未使用分支，增加方程会产生结构差异。
    lax.cos(x)
    return original(x, weight, bias)


def make_inputs(rng):
    """固定签名；每轮改变输入和权重。偏置保证 changed 有数值反例。"""
    return (
        jnp.asarray(rng.normal(size=(3, 4)), dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(4, 2)), dtype=jnp.float32),
        jnp.array([0.5, 1.0], dtype=jnp.float32),
    )


@dataclass(frozen=True)
class Case:
    """预期图标签与数值采样要求分开保存，避免把图不同当成数值必然不同。"""

    name: str
    candidate: Callable
    expected: Status
    same_samples: bool


CASES = (
    Case("独立定义换序", reordered, Status.EQUIVALENT, True),
    Case("加法改成减法", changed, Status.DIFFERENT, False),
    Case("交换加法操作数", swapped_operands, Status.DIFFERENT, True),
    Case("添加未使用分支", unused_branch, Status.DIFFERENT, True),
)


def compile_graph(graph):
    """通过 eval_jaxpr 执行重排图，再用 jit 编译到 CPU，验证实际可执行性。"""
    return jax.jit(lambda *args: core.eval_jaxpr(graph, graph.consts, *args))


def check_values(expected, actual):
    """检查叶子数、shape、dtype 及容差内的数值一致性，不宣称逐位相等。"""
    expected, actual = jax.tree.leaves(expected), jax.tree.leaves(actual)
    assert len(expected) == len(actual), "输出叶子数量改变"
    for a, b in zip(expected, actual, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype, "输出 shape 或 dtype 改变"
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


def run_case(case, samples, show_jaxpr):
    """复核预期图结论、规范化幂等性及各样本重排前后的输出。"""
    # 所有样本使用相同签名，只需追踪一次；输入数值仍在后面的 CPU 执行中逐次变化。
    left = jax.make_jaxpr(original)(*samples[0])
    right = jax.make_jaxpr(case.candidate)(*samples[0])
    result = compare_jaxprs(left, right)
    assert result.status == case.expected, (
        f"预期 {case.expected.value}，实际 {result.status.value}；"
        f"差异={result.difference}；原因={result.issues}"
    )

    normalized = [canonicalize(graph) for graph in (left, right)]
    # 分别从原函数和返回的 jaxpr 构造可执行计算，避免两边误用同一份图而假通过。
    functions = [jax.jit(original), jax.jit(case.candidate)]
    executables = [compile_graph(item.jaxpr) for item in normalized]
    for item in normalized:
        core.check_jaxpr(item.jaxpr)
        again = canonicalize(item.jaxpr)
        assert again.graph == item.graph, "规范化不幂等"
        assert again.jaxpr.eqns == item.jaxpr.eqns, "再次规范化改变方程顺序"

    witness = None
    for inputs in samples:
        a, b = [fn(*inputs) for fn in functions]
        assert jax.tree.structure(a) == jax.tree.structure(b), "函数输出 pytree 不一致"
        # 即使候选与基准不等价，两边各自重排前后的计算也必须保持一致。
        for expected, executable in zip((a, b), executables, strict=True):
            check_values(expected, executable(*jax.tree.leaves(inputs)))
        if case.same_samples:
            check_values(a, b)
        elif witness is None and any(
            not np.array_equal(x, y)
            for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)
        ):
            witness = (inputs, a, b)
    assert case.same_samples or witness is not None, "没有复现预期的数值反例"

    print(f"PASS {case.name}: {result.status.value}；两边重排前后均通过 CPU 检查")
    if result.difference:
        print(f"  图差异位置: {result.difference}")
    if witness is not None:
        inputs, a, b = witness
        print(f"  数值反例输入: {[np.asarray(x).tolist() for x in inputs]}")
        print(f"  基准输出: {jax.tree.map(lambda x: np.asarray(x).tolist(), a)}")
        print(f"  候选输出: {jax.tree.map(lambda x: np.asarray(x).tolist(), b)}")
    if show_jaxpr:
        for label, graph, item in zip(("基准", "候选"), (left, right), normalized, strict=True):
            print(f"  {label}原始 jaxpr:\n{graph}")
            print(f"  BFS 原方程索引={item.bfs_order}，依赖深度={item.depths}")
            print(f"  重排原方程索引={item.equation_order}，complete={item.complete}")
            print(f"  {label}重排 jaxpr:\n{item.jaxpr}")


def positive_int(value):
    """拒绝零次采样，防止根本没有执行数值验证却显示测试通过。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须至少运行一个数值样本")
    return number


def main():
    """固定 CPU、种子和签名；逐个输出结果，并以退出码反映有无失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=positive_int, default=16, help="每个案例的样本数")
    parser.add_argument("--seed", type=int, default=0, help="NumPy 随机种子")
    parser.add_argument("--show-jaxpr", action="store_true", help="显示原图、编号和重排后的图")
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    print(f"JAX {jax.__version__}；CPU；seed={args.seed}；每例 {args.trials} 个样本")
    rng = np.random.default_rng(args.seed)
    samples = [make_inputs(rng) for _ in range(args.trials)]
    failures = 0
    for case in CASES:
        try:
            run_case(case, samples, args.show_jaxpr)
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {case.name}: {exc}")
    print(f"通过 {len(CASES) - failures}/{len(CASES)}；失败时退出码为 1。")
    print("数值采样不证明一般等价；different 仅表示计算图不同。")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
