"""AI 手写的跨原语组合：卷积/矩阵乘/广播与 QR/Cholesky。

改写函数全部固定在源码中。样本生成只负责数值，不生成候选程序。
更广的原语、控制流、通信与真实业务组合由测试矩阵中的其他入口覆盖。
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax


def convolution_dot(x, kernel, weight, bias):
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
    q, r = lax.linalg.qr(x, full_matrices=False)
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    return reconstructed, lower


def factorizations_reordered(x, positive):
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    q, r = lax.linalg.qr(x, full_matrices=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    return reconstructed, lower


def factorizations_interleaved(x, positive):
    q, r = lax.linalg.qr(x, full_matrices=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    return reconstructed, lower


def factorizations_wrong_triangle(x, positive):
    q, r = lax.linalg.qr(x, full_matrices=False)
    lower = lax.linalg.cholesky(positive, symmetrize_input=False)
    reconstructed = lax.dot_general(q, r, (((1,), (0,)), ((), ())))
    return reconstructed, lax.transpose(lower, (1, 0))


@dataclass(frozen=True)
class Case:
    key: str
    programs: tuple
    dtypes: tuple
    primitives: frozenset[str]


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
    rng = np.random.default_rng(6600 + seed)
    if case.key == "conv-dot-broadcast":
        shapes = ((1, 3, 5, 2), (2, 2, 2, 3), (2, 3), (3,))
        return tuple(jnp.asarray(rng.normal(0, 0.3, shape), dtype) for shape in shapes)
    x = rng.normal(0, 0.3, (5, 3))
    a = rng.normal(0, 0.5, (3, 3))
    positive = a @ a.T + np.eye(3)
    return jnp.asarray(x, dtype), jnp.asarray(positive, dtype)


def reference(case, args):
    if case.key == "qr-cholesky":
        x, positive = map(np.asarray, args)
        # QR 重构的独立数学参照是输入本身；不复用 JAX 的 QR 结果。
        return x, np.linalg.cholesky(positive.astype(np.float64)).astype(x.dtype)
    x, kernel, weight, bias = map(np.asarray, args)
    dtype = x.dtype
    padded = np.pad(x.astype(np.float64), ((0, 0), (0, 1), (0, 1), (0, 0)))
    out = np.empty((*x.shape[:3], kernel.shape[-1]), np.float64)
    for row in range(x.shape[1]):
        for col in range(x.shape[2]):
            patch = padded[0, row : row + 2, col : col + 2]
            out[0, row, col] = (patch[..., None] * kernel.astype(np.float64)).sum((0, 1, 2))
    projected = (x.astype(np.float64) @ weight.astype(np.float64)).astype(dtype)
    difference = (out.astype(dtype).astype(np.float64) - projected.astype(np.float64)).astype(dtype)
    return (difference.astype(np.float64) + bias.astype(np.float64)).astype(dtype)
