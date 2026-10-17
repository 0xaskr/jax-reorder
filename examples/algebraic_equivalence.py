"""重放逐元素分配律与矩阵乘法/广播乘加互认，分别报告结构与代数证据。

运行：uv run python examples/algebraic_equivalence.py
两个正例都要求 forward/JVP/VJP 三种模式为 equivalent：分配律依赖逐元素
代数规则，矩阵乘与广播乘加归约依赖同一索引连接。脚本同时打印浮点警告，
说明数学等价只在理想实数解释下成立；CPU 编译只用于确认两侧可执行。
"""

import jax
import jax.numpy as jnp

from jax_reorder import Status, compare_functions


def factored(a, b, c):
    """因式形式 a * (b + c)。"""
    return a * (b + c)


def expanded(a, b, c):
    """分配律展开形式 a * b + a * c。"""
    return a * b + a * c


def matrix(a, b):
    """矩阵乘法基准。"""
    return a @ b


def multiply_sum(a, b):
    """同一索引连接的广播乘法与求和，作为矩阵乘法的改写形式。"""
    return jnp.sum(a[:, :, None] * b[None, :, :], axis=1)


def main():
    """验证 forward/JVP/VJP 并展示数学前提和浮点警告。"""
    vector = jax.ShapeDtypeStruct((4,), jnp.float32)
    # 每项为 (显示名, 基准, 候选, 抽象输入签名)；两个候选都必须实际走代数规则。
    cases = (
        ("分配律", factored, expanded, (vector, vector, vector)),
        (
            "矩阵与乘加归约",
            matrix,
            multiply_sum,
            (jax.ShapeDtypeStruct((2, 3), jnp.float32), jax.ShapeDtypeStruct((3, 4), jnp.float32)),
        ),
    )
    for name, left, right, args in cases:
        # check_cpu 只确认两侧能编译执行；等价结论仍由结构或代数证据给出。
        report = compare_functions(
            left, right, *args, modes=("forward", "jvp", "vjp"), check_cpu=True
        )
        if report.status != Status.EQUIVALENT:
            raise AssertionError(report)
        print(name, report.status.value, report.equivalence_kind, "CPU 编译:", report.cpu_compiled)
        for mode in report.modes:
            print(
                " ",
                mode.mode,
                "结构:",
                mode.comparison.status.value,
                "依据:",
                mode.equivalence_kind,
            )
            if mode.algebra is not None:
                backend = mode.algebra.backend_result
                print(
                    "  后端:",
                    backend.backend if backend else "structural",
                    "工作量:",
                    mode.algebra.work,
                )
        # 两个案例都实际应用了代数规则；该警告说明结论只在理想实数下成立。
        print(" ", report.modes[0].algebra.warnings[0])


if __name__ == "__main__":
    main()
