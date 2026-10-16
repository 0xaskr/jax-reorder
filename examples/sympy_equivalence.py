"""比较 exp(x + y) 与 exp(x) * exp(y)，追踪后直接采用 SymPy 数学结论。

在仓库根目录运行：uv run python examples/sympy_equivalence.py
"""

import jax
import jax.numpy as jnp

from jax_reorder import Status, compare_functions


def original(x, y):
    return jnp.exp(x + y)


def rewritten(x, y):
    return jnp.exp(x) * jnp.exp(y)


def main():
    # 抽象签名足以追踪两份代码，无需构造输入数组或运行数值计算。
    spec = jax.ShapeDtypeStruct((8,), jnp.float32)
    report = compare_functions(original, rewritten, spec, spec, modes=("forward",), check_cpu=False)
    mode = report.modes[0]
    if mode.comparison is None:
        raise AssertionError(report)

    print("计算图比较:", mode.comparison.status.value)
    print("数学等价比较:", mode.status.value)
    print("等价类型:", mode.equivalence_kind)

    if mode.comparison.status != Status.DIFFERENT or mode.status != Status.EQUIVALENT:
        raise AssertionError(report)
    if mode.algebra is None or mode.algebra.backend_result is None:
        raise AssertionError("缺少可信数学后端结果")

    result = mode.algebra.backend_result
    print("数学后端:", result.backend, result.version)
    for query in result.queries:
        print(f"输出 {query.output} / {query.scope}: {query.left} == {query.right}: {query.answer}")
        print("  查询:", query.method)
    print("SymPy 判定通过，查询记录保存在 backend_result。")
    print("数学等价基于理想实数解释，不代表浮点逐位一致。")


if __name__ == "__main__":
    main()
