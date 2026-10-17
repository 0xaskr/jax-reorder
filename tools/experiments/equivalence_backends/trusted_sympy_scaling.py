"""比较显式展开与索引构造两条符号化路径的规模与 Python 内存占用。

运行：`uv run python tools/experiments/equivalence_backends/trusted_sympy_scaling.py`。
对 `sum(x*x)` 在 8/32/128/512 长度分别用 jaxpr_to_sympy_expressions（显式）
与 SymbolicBuilder（索引）构造 SymPy 表达式，统计耗时、Python 峰值内存、
节点数与索引读取数。它只衡量构造开销，不比较证明能力，也不执行 JAX 程序
或评估数值误差；数据只代表本机环境。输出 JSON 到 stdout，退出码 0。
"""

import json
import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from jax_reorder.algebra_explicit import jaxpr_to_sympy_expressions
from jax_reorder.algebra_symbolic import SymbolicBuilder
from jax_reorder.algebra_transport import freeze


def main():
    """运行四组规模 × 两种模式的计时与内存采样，输出 JSON 报告。

    无参数；副作用是初始化 JAX 后端、调用 tracing 与 SymPy 构造，并打印
    报告。不编译或执行追踪出的程序，不写文件。
    """
    # 初始化 JAX 后端；本脚本只 tracing，不编译也不执行生成的计算。
    jax.devices()
    rows = []
    for size in (8, 32, 128, 512):
        graph = jax.make_jaxpr(lambda x: jnp.sum(x * x))(jax.ShapeDtypeStruct((size,), jnp.float32))
        # freeze 得到稳定结构；两种模式消费同一份图，保证输入一致。
        frozen = freeze(graph)
        for mode in ("explicit", "indexed"):
            # tracemalloc 只统计构造期间的 Python 分配，不是进程 RSS。
            tracemalloc.start()
            started = time.perf_counter()
            if mode == "explicit":
                outputs = jaxpr_to_sympy_expressions(graph)[0]
                expression = sp.sympify(np.asarray(outputs[0]).item())
            else:
                # 索引模式保留符号求和/索引，不枚举 size 个元素。
                expression = SymbolicBuilder().graph(frozen)[0].expression
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                {
                    "size": size,
                    "mode": mode,
                    "milliseconds": round(elapsed * 1000, 3),
                    "python_peak_bytes": peak,
                    "unique_nodes": len(set(sp.preorder_traversal(expression))),
                    "indexed_reads": len(expression.atoms(sp.Indexed)),
                    "free_symbols": len(expression.free_symbols),
                }
            )
    print(json.dumps({"jax": jax.__version__, "sympy": sp.__version__, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
