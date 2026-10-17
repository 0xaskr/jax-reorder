"""卷积/矩阵乘/广播与 QR/Cholesky 的固定跨原语组合，供测试矩阵导入。

运行：uv run pytest tests/test_combination_matrix.py -q
本模块没有独立入口。CASES 每个条目的 programs 依次为：基准、独立定义换序、
交错换序（均为正例）和一个改变参数的负例；三个正例必须互相 equivalent，
负例必须 different。sample 只生成数值输入，reference 提供不复用 JAX 分解
结果的 NumPy 参照，二者都不生成候选程序。
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax


def convolution_dot(x, kernel, weight, bias):
    """基准：SAME 卷积与右侧投影互相独立，最后减去投影并加广播偏置。"""
    conv = lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=lax.Precision.HIGH,
    )
    broadcast = lax.broadcast_in_dim(bias, conv.shape, (3,))
    projected = lax.dot_general(x, weight, (((3,), (0,)), ((), ())), precision=lax.Precision.HIGH)
    return (conv - projected) + broadcast


def convolution_dot_reordered(x, kernel, weight, bias):
    """正例：先算投影再算卷积；卷积、减法和加法的操作数位置不变。"""
    projected = lax.dot_general(x, weight, (((3,), (0,)), ((), ())), precision=lax.Precision.HIGH)
    conv = lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=lax.Precision.HIGH,
    )
    broadcast = lax.broadcast_in_dim(bias, conv.shape, (3,))
    return (conv - projected) + broadcast


def convolution_dot_interleaved(x, kernel, weight, bias):
    """正例：偏置广播提前，卷积与投影交错定义，最终表达式完全相同。"""
    broadcast = lax.broadcast_in_dim(bias, (*x.shape[:3], kernel.shape[-1]), (3,))
    projected = lax.dot_general(x, weight, (((3,), (0,)), ((), ())), precision=lax.Precision.HIGH)
    conv = lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=lax.Precision.HIGH,
    )
    return (conv - projected) + broadcast


def convolution_dot_wrong_sign(x, kernel, weight, bias):
    """负例：汇合处由 conv - projected 改成 conv + projected，数值不同。"""
    conv = lax.conv_general_dilated(
        x,
        kernel,
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=lax.Precision.HIGH,
    )
    projected = lax.dot_general(x, weight, (((3,), (0,)), ((), ())), precision=lax.Precision.HIGH)
    broadcast = lax.broadcast_in_dim(bias, conv.shape, (3,))
    return (conv + projected) + broadcast


def factorizations(x, positive):
    """基准：QR 重构与 Cholesky 分解独立，返回 (重构, 下三角)。"""
    q, r = lax.linalg.qr(x, full_matrices=False)
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    return reconstructed, lower


def factorizations_reordered(x, positive):
    """正例：先做 Cholesky 再做 QR，两个输出端口顺序保持。"""
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    q, r = lax.linalg.qr(x, full_matrices=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    return reconstructed, lower


def factorizations_interleaved(x, positive):
    """正例：QR、重构、Cholesky 交错定义，数据依赖与端口顺序不变。"""
    q, r = lax.linalg.qr(x, full_matrices=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    return reconstructed, lower


def factorizations_wrong_triangle(x, positive):
    """负例：第二个输出返回 upper 而非 lower 三角，端口含义已改变。"""
    q, r = lax.linalg.qr(x, full_matrices=False)
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    return reconstructed, lax.transpose(lower, (1, 0))


@dataclass(frozen=True)
class Case:
    """一组固定组合：programs 以基准开头、负例结尾，dtypes 为参与追踪的类型。"""

    key: str
    programs: tuple
    dtypes: tuple
    primitives: frozenset[str]


# 两个组合族；primitives 用于断言目标原语确实出现在基准图里。
CASES = (
    Case(
        "conv-dot-broadcast",
        (
            convolution_dot,
            convolution_dot_reordered,
            convolution_dot_interleaved,
            convolution_dot_wrong_sign,
        ),
        (jnp.float32, jnp.float64, jnp.bfloat16),
        frozenset(("conv_general_dilated", "dot_general", "broadcast_in_dim")),
    ),
    Case(
        "qr-cholesky",
        (
            factorizations,
            factorizations_reordered,
            factorizations_interleaved,
            factorizations_wrong_triangle,
        ),
        (jnp.float32, jnp.float64),
        frozenset(("qr", "cholesky", "dot_general")),
    ),
)


def sample(case, dtype, seed):
    """生成固定形状的随机输入；seed 只改变数值，不参与候选或标签的构造。"""
    rng = np.random.default_rng(6600 + seed)
    if case.key == "conv-dot-broadcast":
        shapes = ((1, 3, 5, 2), (2, 2, 2, 3), (2, 3), (3,))
        return tuple(jnp.asarray(rng.normal(0, 0.3, shape), dtype) for shape in shapes)
    x = rng.normal(0, 0.3, (5, 3))
    a = rng.normal(0, 0.5, (3, 3))
    positive = a @ a.T + np.eye(3)
    return jnp.asarray(x, dtype), jnp.asarray(positive, dtype)


def reference(case, args):
    """独立 NumPy 参照：不复用 JAX 的分解、卷积或矩阵乘法结果。"""
    if case.key == "qr-cholesky":
        x, positive = map(np.asarray, args)
        # QR 重构的独立数学参照是输入本身；不复用 JAX 的 QR 结果。
        return x, np.linalg.cholesky(positive.astype(np.float64)).astype(x.dtype)
    x, kernel, weight, bias = map(np.asarray, args)
    dtype = x.dtype
    # 手工实现 SAME 卷积：按 (1,1) 步长、2x2 窗口、NHWC 布局补零后逐点累加。
    padded = np.pad(x.astype(np.float64), ((0, 0), (0, 1), (0, 1), (0, 0)))
    out = np.empty((*x.shape[:3], kernel.shape[-1]), np.float64)
    for row in range(x.shape[1]):
        for col in range(x.shape[2]):
            patch = padded[0, row : row + 2, col : col + 2]
            out[0, row, col] = (patch[..., None] * kernel.astype(np.float64)).sum((0, 1, 2))
    projected = (x.astype(np.float64) @ weight.astype(np.float64)).astype(dtype)
    difference = (out.astype(dtype).astype(np.float64) - projected.astype(np.float64)).astype(dtype)
    return (difference.astype(np.float64) + bias.astype(np.float64)).astype(dtype)
