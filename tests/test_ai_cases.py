"""回放预标注的 AI 改写用例，并用独立 CPU 执行核对标签。

覆盖：
- 正例：tests/bfs_reordering.py 的 ai_cases 固定 18 个用例（换序、代数恒等以及应判
  different 的改写），由 verify_case 复核预期标签；每个 case 两侧的 jaxpr 都重新 jit
  执行，与原始 Python 函数按 rtol/atol=1e-6 数值对照，并要求规范化幂等。
- 手动构造的 Llama MLP：非平凡分片 shape 上只交换独立投影的定义顺序，比较 equivalent
  且两个顺序的数值结果逐位相等。
- 反例：数值近似不能证明图等价；标签本身只是待验证的预期，由 verify_case 独立复核。

前置条件：model_mesh() 需要 2 个 CPU 设备（conftest.py）；固定种子 17 只影响数值对照，
不改变被比较的图。18 个固定用例不覆盖所有 AI 改写。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.bfs_reordering import ai_cases, llama_mlp, model_mesh, verify_case
from jax_reorder import Status, canonicalize, compare_jaxprs


@pytest.mark.parametrize("index", range(18))
def test_saved_ai_case(index):
    """逐个回放固定 AI 用例：标签、可执行性、数值对照和规范化幂等一起检查。"""
    mesh = model_mesh()
    with jax.set_mesh(mesh):
        case = list(ai_cases(mesh))[index]
        comparison = verify_case(case)
        # 两侧都通过 eval_jaxpr 实际执行，而不只是相信比较结果。
        for graph, fn in ((comparison.left, case.base), (comparison.right, case.candidate)):
            actual = jax.jit(
                lambda *xs, graph=graph: core.eval_jaxpr(graph.jaxpr, graph.jaxpr.consts, *xs)
            )(*jax.tree.leaves(case.args))
            for a, b in zip(actual, jax.tree.leaves(fn(*case.args)), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
            # 规范化必须幂等：再次规范化得到的图与第一次相同。
            assert canonicalize(graph.jaxpr).graph == graph.graph


def test_llama_layer_with_nontrivial_shapes():
    """分片 Llama MLP 的独立投影换序：结构等价，且两个顺序数值逐位相同。"""
    mesh = model_mesh()
    rng = np.random.default_rng(17)
    with jax.set_mesh(mesh):
        # 每个输入按显式 NamedSharding 放置，模拟真实分片模型的参数布局。
        args = tuple(
            jax.device_put(
                jnp.asarray(rng.normal(size=shape), dtype=jnp.float32), NamedSharding(mesh, spec)
            )
            for shape, spec in (
                ((2, 3), P("data", None)),
                ((3, 4), P(None, "tensor")),
                ((3, 4), P(None, "tensor")),
                ((4, 3), P("tensor", None)),
            )
        )
        left = jax.make_jaxpr(lambda *xs: llama_mlp(*xs, mesh=mesh))(*args)
        # reverse=True 只交换两个独立投影的顺序，不改变任何依赖或数值运算。
        right = jax.make_jaxpr(lambda *xs: llama_mlp(*xs, mesh=mesh, reverse=True))(*args)
        result = compare_jaxprs(left, right)
        assert result.status == Status.EQUIVALENT, result
        a = llama_mlp(*args, mesh=mesh)
        b = llama_mlp(*args, mesh=mesh, reverse=True)
        # 两次调用执行的是同一组浮点运算，只是发射顺序不同，因此可以要求逐位相等。
        np.testing.assert_array_equal(a, b)
