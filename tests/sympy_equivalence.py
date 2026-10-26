"""比较 exp(x + y) 与 exp(x) * exp(y)，追踪后直接采用 SymPy 数学结论。

在仓库根目录运行：uv run python tests/sympy_equivalence.py
预期结构比较为 different（两份代码的计算图不同），数学比较为 equivalent
（信赖 SymPy 的实数恒等式判定）。脚本只做抽象追踪，不编译、不执行数值；
结论限定在理想实数解释，不表示浮点逐位一致或误差很小。
"""

import jax
import jax.numpy as jnp

from jax_reorder import Status, compare_functions


def original(x, y):
    """基准：只做一次 exp(x + y)。"""
    return jnp.exp(x + y)


def rewritten(x, y):
    """候选：拆成 exp(x) * exp(y)，数学等价但计算图不同。"""
    return jnp.exp(x) * jnp.exp(y)


def main():
    """断言结构不同、数学等价，并打印后端版本与逐输出查询记录。"""
    # 抽象签名足以追踪两份代码，无需构造输入数组或运行数值计算。
    spec = jax.ShapeDtypeStruct((8,), jnp.float32)
    report = compare_functions(original, rewritten, spec, spec, modes=("forward",), check_cpu=False)
    mode = report.modes[0]
    if mode.comparison is None:
        raise AssertionError(report)

    print("计算图比较:", mode.comparison.status.value)
    print("数学等价比较:", mode.status.value)
    print("等价类型:", mode.equivalence_kind)

    # 结构不同是预期结果；只有数学后端给出约定结论才算通过，防止静默降级为结构比较。
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
