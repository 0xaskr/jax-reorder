"""重放逐元素分配律与矩阵乘法/广播乘加互认，分别报告结构与代数证据。"""

import jax
import jax.numpy as jnp

from jax_reorder import Status, compare_functions, verify_algebraic_proof


def factored(a, b, c):
    """因式形式。"""
    return a * (b + c)


def expanded(a, b, c):
    """分配律展开形式。"""
    return a * b + a * c


def matrix(a, b):
    """矩阵乘法。"""
    return a @ b


def multiply_sum(a, b):
    """同一索引连接的广播乘法与求和。"""
    return jnp.sum(a[:, :, None] * b[None, :, :], axis=1)


def main():
    """验证 forward/JVP/VJP 并展示数学前提和浮点警告。"""
    vector = jax.ShapeDtypeStruct((4,), jnp.float32)
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
                verify_algebraic_proof(mode.left.jaxpr, mode.right.jaxpr, mode.algebra.proof)
                print("  工作量:", mode.algebra.work)
        print(" ", report.modes[0].algebra.warnings[0])


if __name__ == "__main__":
    main()
