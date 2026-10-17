"""展示独立运算换序前后的 jaxpr、BFS 编号和图比较结果。

运行：uv run python examples/inspect_jaxpr.py
两个函数保持 add 的操作数顺序，只交换 sin 与 cos 的定义位置。脚本只做
抽象追踪和规范化，不执行数值、不编译。打印的 BFS 编号和方程索引是各自
原图的局部信息，等价结论以最后一行图比较状态为准，不能由索引列表直接读出。
"""

import jax
import jax.numpy as jnp

from jax_reorder import __version__, canonicalize, compare_jaxprs


def first(x, y):
    """基准：先定义 sin 分支，再定义 cos 分支。"""
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def second(x, y):
    """候选：两个分支互不依赖，交换定义顺序后应仍得到 equivalent。"""
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b


def main():
    """按固定抽象签名追踪两份代码，打印编号、重排图和最终图比较状态。"""
    x = jax.ShapeDtypeStruct((4,), jnp.float32)
    y = jax.ShapeDtypeStruct((4,), jnp.float32)
    print(f"jax-reorder {__version__}; JAX {jax.__version__}")
    print("First trace:")
    left = jax.make_jaxpr(first)(x, y)
    print(left)
    print("Second trace:")
    right = jax.make_jaxpr(second)(x, y)
    print(right)
    for label, graph in (("First", left), ("Second", right)):
        result = canonicalize(graph)
        # 这两个元组记录的是各自原图的方程索引，本身不要求两边数值相同。
        # 是否图等价要比较编号后的节点、引用和语义字段，而非直接比较索引列表。
        print(f"{label}: BFS={result.bfs_order}, equation order={result.equation_order}")
        print(result.jaxpr)
    print(f"Graph comparison: {compare_jaxprs(left, right).status.value}")


if __name__ == "__main__":
    main()
