# Array mappings and explicit traversal adapted from johnviljoen/jax2sympy,
# revision 2cb6a6568716b26af0abd542a941f87ae0580e25.
# MIT License
#
# Copyright (c) 2026 John Viljoen
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""逐元素 JAX→SymPy 转换的参考实现，保留给公式基线与后续适配器。

本文件把完整图展开成 NumPy 对象数组，每个元素对应一个 SymPy 表达式；
它不跟踪定义条件、有限性或资源预算，因此不参与生产数学判等。
生产路径使用 SymbolicBuilder：惰性 Indexed/Sum 表示、原始定义条件和预算。
这里的映射与遍历改自 jax2sympy 的固定 revision（来源与许可见上方文件头），
未验证的映射在下方逐项标注；新增 primitive 前需要补数值回归。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

# 保留上游快照的 lambda 赋值和宽松循环写法，以最小差异固定来源版本。
# ruff: noqa: E731, B007, B905, E741
import jax
import numpy as np
import sympy as sy
from jax import lax
from jax._src import core
from jax.extend.core import Literal

from ._typing import ObjectArray, Shape
from .expression import _exact_array as exact_array

type SymbolicArray = ObjectArray | sy.Basic


# 逐点映射：对象数组上的二元运算可直接用 Python 运算符完成；
# 一元超越函数用 np.vectorize 包装，otypes=[object] 阻止 NumPy 推断浮点 dtype。
_sym_add: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: a + b
_sym_sub: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: a - b
_sym_mul: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: a * b
_sym_div: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: a / b
_sym_pow: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: a**b
_sym_integer_pow: Callable[[ObjectArray, int], ObjectArray] = lambda a, exp: a**exp
_sym_neg: Callable[[ObjectArray], ObjectArray] = lambda a: -a
_sym_exp: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.exp, otypes=[object])(a)
_sym_log: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.log)(a)
_sym_sin: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.sin)(a)
_sym_cos: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.cos)(a)
_sym_tan: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.tan)(a)
_sym_asin: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.asin)(a)
_sym_acos: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.acos)(a)
_sym_atan: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.atan)(a)
_sym_atan2: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.atan2)(
    a, b
)  # 未验证（2026-07-14 加入，尚无数值回归覆盖）
_sym_sinh: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.sinh)(a)
_sym_cosh: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.cosh)(a)
_sym_tanh: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.tanh)(a)
_sym_sqrt: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.sqrt)(a)
_sym_sign: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.sign)(a)
_sym_square: Callable[[ObjectArray], ObjectArray] = lambda a: a**2
_sym_abs: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.Abs)(a)
# 比较与逻辑关系必须用 Eq/Ne/And 等构造：Python 的 == 和 < 会返回 bool，
# 而不是可参与符号运算的关系表达式。sy.And/Or/Not 是函数，需逐元素调用。
_sym_eq: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.Eq)(a, b)
_sym_ne: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.Ne)(a, b)
_sym_lt: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(
    sy.StrictLessThan
)(a, b)
_sym_le: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.LessThan)(
    a, b
)
_sym_gt: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(
    sy.StrictGreaterThan
)(a, b)
_sym_ge: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(
    sy.GreaterThan
)(a, b)
_sym_and: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.And)(
    a, b
)
_sym_or: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.Or)(a, b)
_sym_not: Callable[[ObjectArray], ObjectArray] = lambda a: np.vectorize(sy.Not)(a)
_sym_max: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.Max)(
    a, b
)
_sym_min: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(sy.Min)(
    a, b
)
_sym_logaddexp: Callable[[ObjectArray, ObjectArray], ObjectArray] = lambda a, b: np.vectorize(
    lambda x, y: sy.log(sy.exp(x) + sy.exp(y))
)(a, b)
# logaddexp 按定义展开成 log(exp(a)+exp(b))，不引入新的不透明函数符号。
# 该公式在极端输入下与稳定的 JAX 实现数值不同，但作为公式基线足够。


# 数组级映射：下面的映射直接作用在 NumPy 对象数组上，不构造逐元素循环；
# 依赖 JAX 静态参数（axes、dimension_numbers 等）确定轴映射。
def _sym_reduce_sum(a: Sequence[ObjectArray], eqn: core.JaxprEqn) -> SymbolicArray:
    """沿静态 axes 对单个操作数求和；多操作数输入由断言拒绝。"""
    assert len(a) == 1
    return np.sum(a[0], axis=eqn.params["axes"])


def _sym_reduce_or(a: Sequence[ObjectArray], eqn: core.JaxprEqn) -> SymbolicArray:
    """布尔归约：axes 为空时把所有元素合成一个 Or，否则只沿 axes[0] 应用。"""
    assert len(a) == 1
    arr = a[0]
    axes = eqn.params["axes"]
    if axes == ():
        # 归约所有轴：把所有元素合成一个 Or。
        return sy.Or(*arr.flat)
    else:
        # 注意：只应用 axes[0]，多轴归约的其余轴会被忽略；该路径未做数值
        # 回归，生产转换不依赖此映射。
        return np.apply_along_axis(lambda x: sy.Or(*x), axes[0], arr)


def _sym_transpose(a: ObjectArray, eqn: core.JaxprEqn) -> ObjectArray:
    """按置换转置数组；置换参数的三种键名来自不同接口的命名约定。"""
    # 未验证的上游路径；生产转换不依赖此映射。
    perm = (
        eqn.params.get("permutation")  # JAX 规范拼写
        or eqn.params.get("axes")  # NumPy 命名
        or eqn.params.get("dims")  # 更早的内部拼写
    )
    if perm is None:
        # 未给出置换时按反序轴转置，与 NumPy 默认行为一致。
        perm = tuple(range(a.ndim))[::-1]
    return np.transpose(a, axes=perm)


# add_any 只表示“把第二个操作数加到第一个”的纯运算，可直接用加法。
def _sym_add_any(inexprs: Sequence[ObjectArray]) -> SymbolicArray:
    """把多个操作数依次相加；利用 NumPy 广播对齐槽位形状。"""
    # 依赖广播按顺序累加，不假设各操作数形状完全相同。
    result = inexprs[0]
    for array in inexprs[1:]:
        result = np.add(result, array)  # np.add 负责广播
    return result


def _sym_select_n(inexprs: Sequence[ObjectArray]) -> ObjectArray:
    """逐元素选择：布尔谓词走 np.where，符号谓词构造 Piecewise。

    JAX 的 select_n_p 操作数顺序是（谓词，假分支，真分支）；符号谓词无法
    预先求值，只能对每个元素建立 Piecewise 分支。
    """
    pred, false_val, true_val = inexprs

    # 真实布尔数组可直接交给 NumPy，避免为每个元素构造 SymPy 分支。
    if pred.dtype != object and pred.dtype == bool:
        return np.where(pred, true_val, false_val)

    # 通用符号路径：逐元素 Piecewise；该路径未做数值回归，生产转换不依赖。
    piecewise_fn = np.vectorize(
        lambda p, t, f: sy.Piecewise((t, p), (f, True)),
        otypes=[object],
    )
    return piecewise_fn(pred, true_val, false_val)


def _sym_pad(inexprs: Sequence[ObjectArray], eqn: core.JaxprEqn) -> ObjectArray:
    """按 padding_config 填充常量数组；interior 是相邻元素之间的填充数。"""
    assert len(inexprs) == 2

    inexpr = inexprs[0]
    padding_value = inexprs[1]
    padding_config = eqn.params["padding_config"]

    # 每个轴：low + (n-1)*(interior+1) + 1 + high，含内部间隔后的总长。
    padded_shape = [
        low + (inexpr.shape[i] - 1) * (interior + 1) + 1 + high
        for i, (low, high, interior) in enumerate(padding_config)
    ]

    # 先铺满填充值，再把原数组写到对应切片；dtype 跟随操作数。
    result = np.full(padded_shape, padding_value, dtype=inexpr.dtype)

    # 原数组元素按 interior+1 的步长写入，两端留出 low/high 个填充。
    insert_slices = tuple(
        slice(low, low + inexpr.shape[i] * (interior + 1), interior + 1)
        for i, (low, high, interior) in enumerate(padding_config)
    )

    # 把操作数放入填充结果。
    result[insert_slices] = inexpr
    return result


# 类型转换在逐元素公式里视为恒等；dtype 舍入/溢出语义由生产构造器处理。
_sym_convert_element_type: Callable[[ObjectArray], ObjectArray] = lambda a: a


def _sym_reshape(a: ObjectArray, new_sizes: Shape, dimensions: Sequence[int] | None) -> ObjectArray:
    """按 new_sizes 重排数组；只支持无轴置换的 C 顺序 reshape。

    dimensions 为 None 或含 0 时直接 reshape；其余置换未实现，抛
    NotImplementedError 而不是给出错误公式。注意该条件并不验证含 0 的
    dimensions 是否就是恒等置换。
    """
    if dimensions is None or 0 in dimensions:
        a = a.reshape(*new_sizes)
    else:
        raise NotImplementedError
    return a


def _sym_squeeze(a: ObjectArray, dimensions: Sequence[int]) -> ObjectArray:
    """删除 dimensions 指定的单位轴，其余轴保留。"""
    slices = [0 if i in dimensions else slice(None) for i in range(a.ndim)]
    return a[tuple(slices)]


def _sym_iota(eqn: core.JaxprEqn) -> ObjectArray:
    """构造 iota 常量数组：只有指定轴递增，其余轴广播同一序列。"""
    shape = eqn.params["shape"]
    dimension = eqn.params["dimension"]
    return (
        np.arange(shape[dimension])
        .reshape([1 if i != dimension else shape[dimension] for i in range(len(shape))])
        .repeat(np.prod(shape) // shape[dimension], axis=dimension)
        .reshape(shape)
    )


def _sym_split(inexprs: Sequence[ObjectArray], eqn: core.JaxprEqn) -> list[ObjectArray]:
    """沿静态 axis 把单个操作数切成多段，输出顺序与 sizes 一致。"""
    sizes = eqn.params["sizes"]
    axis = eqn.params["axis"]
    assert len(inexprs) == 1
    operand = inexprs[0]
    # 累积各段长度得到切分点；该参考实现未做数值回归。
    indices = np.cumsum(sizes[:-1])
    return np.split(operand, indices, axis=axis)


def _sym_slice(
    a: ObjectArray,
    start_indices: Sequence[int],
    limit_indices: Sequence[int],
    strides: Sequence[int] | None,
) -> ObjectArray:
    """按静态起止与步长取切片；strides 缺省时按 1 处理。"""
    axis_slice = []
    if strides is not None:
        for start, limit, stride in zip(start_indices, limit_indices, strides):
            axis_slice.append([start, limit, stride])
    else:
        for start, limit in zip(start_indices, limit_indices):
            axis_slice.append([start, limit, 1])
    return a[tuple([slice(s[0], s[1], s[2]) for s in axis_slice])]


def _sym_dynamic_slice(inexprs: Sequence[ObjectArray], eqn: core.JaxprEqn) -> ObjectArray:
    """按运行时起始下标与静态 slice_sizes 取切片。"""
    # inexprs[0] 是被切数组，inexprs[1:] 是各轴起始下标（标量或单元素数组）。
    a = inexprs[0]
    start_indices = inexprs[1:]
    slice_sizes = eqn.params["slice_sizes"]
    # 构造 slice(start[i], start[i] + size[i])；起始下标必须是可取的整数。
    slices = tuple(
        slice(
            int(start.flat[0]) if hasattr(start, "flat") else int(start),
            int(start.flat[0]) + size if hasattr(start, "flat") else int(start) + size,
        )
        for start, size in zip(start_indices, slice_sizes)
    )
    return a[slices]


def _sym_dot_general(a: ObjectArray, b: ObjectArray, eqn: core.JaxprEqn) -> SymbolicArray:
    """把 dot_general 的批次/收缩/自由轴映射成 einsum 轴序列。"""
    (ca, cb), (ba, bb) = eqn.params["dimension_numbers"]
    # la/lb 是两侧操作数在 einsum 中的占位轴号，右操作数接在左操作数之后。
    la, lb = list(range(a.ndim)), list(range(a.ndim, a.ndim + b.ndim))
    # 收缩轴与批次轴共享同一下标后，einsum 自然完成乘加。
    for left, right in zip(ca + ba, cb + bb, strict=True):
        lb[right] = la[left]
    # 输出轴：批次轴、左自由轴、右自由轴，按输出位置排列。
    output = [la[i] for i in ba]
    output += [la[i] for i in range(a.ndim) if i not in ca + ba]
    output += [lb[i] for i in range(b.ndim) if i not in cb + bb]
    return np.einsum(a, la, b, lb, output)


def _sym_reduce_extreme(
    a: Sequence[ObjectArray], eqn: core.JaxprEqn, function: Callable[..., sy.Basic]
) -> SymbolicArray:
    """沿 axes 依次求最大值或最小值；从大到小归约避免轴号偏移。"""
    result = a[0]
    for axis in sorted(eqn.params["axes"], reverse=True):
        result = np.apply_along_axis(lambda row: function(*row), axis, result)
    return result


def _sym_gather(
    operand: ObjectArray,
    start_indices: ObjectArray,
    dimension_numbers: lax.GatherDimensionNumbers,
    slice_sizes: Sequence[int],
    mode: lax.GatherScatterMode,
) -> ObjectArray:
    """按 gather 的窗口/批次维描述逐元素取数；仅支持一维 operand。

    该参考实现断言 offset_dims 与 operand_rank 均为一维，其他布局直接
    AssertionError，而不是给出可能错误的公式。CLIP 与 PROMISE_IN_BOUNDS
    按各自模式处理，其余模式做严格越界检查，越界抛 IndexError。
    """
    # 解包维度描述；当前只支持一个 offset 维。
    offset_dims = dimension_numbers.offset_dims
    assert len(offset_dims) == 1
    collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
    start_index_map = dimension_numbers.start_index_map

    # index_vector_dim 未单独提供时假定为起始下标的最后一维。
    index_vector_dim = len(start_indices.shape) - 1

    # 输出形状 = 批次形状 + 未被折叠的窗口形状。
    batch_dims = [dim for dim in range(len(start_indices.shape)) if dim != index_vector_dim]
    batch_shape = [start_indices.shape[d] for d in batch_dims]
    offset_shape = [
        slice_sizes[dim] for dim in range(len(slice_sizes)) if dim not in collapsed_slice_dims
    ]
    output_shape = batch_shape + offset_shape
    output = np.empty(output_shape, dtype=operand.dtype)

    # 预计算 offset 维到 operand 可见维的映射。
    remapped_offset_dims = []
    operand_rank = len(operand.shape)
    assert operand_rank == 1
    available_dims = [dim for dim in range(operand_rank) if dim not in collapsed_slice_dims]
    assert len(offset_dims) == 1
    for idx, offset_dim in enumerate(offset_dims):
        remapped_offset_dims.append(available_dims[idx])

    # 逐输出元素重建索引；输出规模由静态 shape 决定。
    for out_index in np.ndindex(*output_shape):
        # 步骤 1：从输出下标取出批次下标 G。
        G = [out_index[d] for d in range(len(batch_dims))]

        # 步骤 2：用 start_indices 计算起始下标 S。
        S: list[int | ObjectArray] = [0] * len(start_index_map)
        for i, map_dim in enumerate(start_index_map):
            combined_index = tuple(G[:index_vector_dim] + [i] + G[index_vector_dim:])
            S[i] = start_indices[combined_index]

        # 步骤 3：把 S 投影回 operand 轴得到 Sin。
        Sin: list[int | ObjectArray] = [0] * operand_rank
        for i, map_dim in enumerate(start_index_map):
            Sin[map_dim] = S[i]

        # 步骤 4：把输出 offset 投影回 operand 轴得到 Oin。
        Oin = [0] * operand_rank
        for i, offset_dim in enumerate(offset_dims):
            Oin[remapped_offset_dims[i]] = out_index[len(batch_dims) + i]

        # 步骤 5：最终读取坐标 In = Sin + Oin；一维特化再做一次偏移改写。
        In = tuple(Sin[d] + Oin[d] for d in range(operand_rank))
        In = (In[0] - offset_dim * operand.shape[0],)  # 上游快照的一维偏移改写

        # 步骤 6：按 mode 处理越界下标。
        if mode == jax.lax.GatherScatterMode.CLIP:
            In = tuple(max(0, min(idx, dim - 1)) for idx, dim in zip(In, operand.shape))
        elif mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            pass  # 调用方承诺下标合法，不加额外检查。
        else:
            if any(idx < 0 or idx >= dim for idx, dim in zip(In, operand.shape)):
                raise IndexError(f"Out-of-bounds index: {In}")

        # 步骤 7：写入输出数组。
        output[out_index] = operand[In]

    return output


def _sym_scatter(
    operand: ObjectArray,
    scatter_indices: ObjectArray,
    updates: ObjectArray,
    dimension_numbers: lax.ScatterDimensionNumbers,
    mode: lax.GatherScatterMode,
    combine: Callable[[Any, Any], Any] | None = None,
) -> ObjectArray:
    """按 scatter 描述把 updates 写入 operand 副本，返回新数组。

    该参考实现按上游快照处理窗口/插入维映射，没有额外数值回归。
    combine=None 表示 lax.scatter 的覆盖语义；scatter-add 传入加法合并，
    使落在同一坐标的更新累加。越界行为由 mode 决定。
    """
    # 解包维度描述。
    update_window_dims = dimension_numbers.update_window_dims
    inserted_window_dims = dimension_numbers.inserted_window_dims
    scatter_dims_to_operand_dims = dimension_numbers.scatter_dims_to_operand_dims

    # 从 operand 副本开始，未命中坐标保持原值。
    output = operand.copy()

    # update_scatter_dims 是 updates 中不参与窗口的轴，用于取批次下标。
    update_scatter_dims = [i for i in range(updates.ndim) if i not in update_window_dims]

    def build_window_dims_to_operand_dims(
        update_window_dims: Sequence[int], inserted_window_dims: Sequence[int], operand_rank: int
    ) -> dict[int, int]:
        """建立 update_window_dims 到 operand 维的映射，跳过插入维。"""
        # 先取 operand 中所有未被插入的轴，再按顺序对应窗口维。
        available_dims = [i for i in range(operand_rank) if i not in inserted_window_dims]
        mapping = {dim: available_dims[idx] for idx, dim in enumerate(update_window_dims)}
        return mapping

    # 逐个更新元素计算目标坐标并写入。
    for update_index in np.ndindex(*updates.shape):
        # 步骤 1：从 update 下标取出批次下标 G。
        G = [update_index[dim] for dim in update_scatter_dims]

        # 步骤 2：用 scatter_indices 计算目标起始下标 S。
        S: list[int | ObjectArray] = [0] * len(scatter_dims_to_operand_dims)
        for i, scatter_dim in enumerate(scatter_dims_to_operand_dims):
            # 把 G 与当前维度序号组合成读取 scatter_indices 的下标。
            combined = tuple(G[:scatter_dim] + [i] + G[scatter_dim:])
            S[i] = scatter_indices[combined]

        # 步骤 3：把 S 投影回 operand 轴得到 Sin。
        Sin: list[int | ObjectArray] = [0] * operand.ndim
        for i, scatter_dim in enumerate(scatter_dims_to_operand_dims):
            Sin[scatter_dim] = S[i]

        # 步骤 4：把 update 窗口下标映射到 operand 轴得到 Win。
        Win = [0] * operand.ndim
        window_mapping = build_window_dims_to_operand_dims(
            update_window_dims, inserted_window_dims, operand.ndim
        )
        for i in update_window_dims:
            Win[window_mapping[i]] = update_index[i]

        # 步骤 5：目标坐标 I = Win + Sin。
        I = tuple(int(Win[d] + Sin[d]) for d in range(operand.ndim))

        # 步骤 6：按 mode 处理越界下标。
        if mode == jax.lax.GatherScatterMode.CLIP:
            I = tuple(max(0, min(idx, dim - 1)) for idx, dim in zip(I, operand.shape))
        elif mode == jax.lax.GatherScatterMode.FILL_OR_DROP:
            if any(idx < 0 or idx >= dim for idx, dim in zip(I, operand.shape)):
                continue  # 越界更新被丢弃，副本保持原值。
        elif mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            pass  # 调用方承诺下标合法。
        else:
            if any(idx < 0 or idx >= dim for idx, dim in zip(I, operand.shape)):
                raise IndexError(f"Out-of-bounds index: {I}")

        # 步骤 7：应用更新；combine 非空时与旧值合并。
        if combine is None:
            output[I] = updates[update_index]
        else:
            output[I] = combine(output[I], updates[update_index])

    return output


def _sym_broadcast_in_dim(
    a: Sequence[ObjectArray], shape: Shape, broadcast_dimensions: Sequence[int]
) -> ObjectArray:
    """把单操作数按 broadcast_dimensions 广播到目标 shape。"""
    assert len(a) == 1
    operand = a[0]
    # 先按 broadcast_dimensions 插入单位轴，再交给 NumPy 广播到完整形状。
    reshaped_shape = [1] * len(shape)
    for i, dim in enumerate(broadcast_dimensions):
        reshaped_shape[dim] = operand.shape[i]
    a_reshaped = np.reshape(operand, reshaped_shape)
    return np.broadcast_to(a_reshaped, shape)


def _sym_concatenate(arr_list: Sequence[ObjectArray], dimension: int = 0) -> ObjectArray:
    """沿静态 dimension 拼接多个数组，不插入新轴。"""
    return np.concat(arr_list, axis=dimension)


def _sym_stack(arr_list: Sequence[ObjectArray], axis: int = 0) -> ObjectArray:
    """沿新轴 axis 堆叠数组。"""
    # jax >= 0.9 引入一等的 stack 原语（jnp.stack 不再降为 concatenate+reshape）。
    # 上游注释提到的 test_jax_upgrade_surface.py 不在本仓库，生产转换不走此映射。
    return np.stack(arr_list, axis=axis)


def _sym_cumsum(a: Sequence[ObjectArray], eqn: core.JaxprEqn) -> ObjectArray:
    """前缀和；reverse=True 时先翻转再累加，最后翻回原顺序。"""
    # 未验证（2026-07-14 加入，尚无数值回归覆盖）。
    assert len(a) == 1
    arr = a[0]
    axis = eqn.params["axis"]
    reverse = eqn.params.get("reverse", False)
    if reverse:
        arr = np.flip(arr, axis=axis)
    out = np.cumsum(arr, axis=axis)
    return np.flip(out, axis=axis) if reverse else out


# primitive 名称 → 逐元素公式。值接收 (inexprs, eqn)，返回单个数组或按
# outvars 顺序排列的 list（多输出原语）。未列出的原语由 get_sym_outvar
# 抛 NotImplementedError，绝不返回占位表达式。
primitive_to_sympy_op: dict[
    str, Callable[[Sequence[ObjectArray], core.JaxprEqn], SymbolicArray | list[ObjectArray]]
] = {
    "add": lambda inexprs, eqn: _sym_add(*inexprs),
    "sub": lambda inexprs, eqn: _sym_sub(*inexprs),
    "mul": lambda inexprs, eqn: _sym_mul(*inexprs),
    "div": lambda inexprs, eqn: _sym_div(*inexprs),
    "neg": lambda inexprs, eqn: _sym_neg(*inexprs),
    "pow": lambda inexprs, eqn: _sym_pow(*inexprs),
    "integer_pow": lambda inexprs, eqn: _sym_integer_pow(inexprs[0], eqn.params["y"]),
    "sign": lambda inexprs, eqn: _sym_sign(*inexprs),
    "square": lambda inexprs, eqn: _sym_square(*inexprs),
    "abs": lambda inexprs, eqn: _sym_abs(*inexprs),
    "exp": lambda inexprs, eqn: _sym_exp(*inexprs),
    "log": lambda inexprs, eqn: _sym_log(*inexprs),
    "sin": lambda inexprs, eqn: _sym_sin(*inexprs),
    "cos": lambda inexprs, eqn: _sym_cos(*inexprs),
    "tan": lambda inexprs, eqn: _sym_tan(*inexprs),
    "asin": lambda inexprs, eqn: _sym_asin(*inexprs),
    "acos": lambda inexprs, eqn: _sym_acos(*inexprs),
    "atan": lambda inexprs, eqn: _sym_atan(*inexprs),
    "sinh": lambda inexprs, eqn: _sym_sinh(*inexprs),
    "cosh": lambda inexprs, eqn: _sym_cosh(*inexprs),
    "tanh": lambda inexprs, eqn: _sym_tanh(*inexprs),
    "sqrt": lambda inexprs, eqn: _sym_sqrt(*inexprs),
    "eq": lambda inexprs, eqn: _sym_eq(*inexprs),
    "ne": lambda inexprs, eqn: _sym_ne(*inexprs),
    "lt": lambda inexprs, eqn: _sym_lt(*inexprs),
    "le": lambda inexprs, eqn: _sym_le(*inexprs),
    "gt": lambda inexprs, eqn: _sym_gt(*inexprs),
    "ge": lambda inexprs, eqn: _sym_ge(*inexprs),
    "not": lambda inexprs, eqn: _sym_not(*inexprs),
    "and": lambda inexprs, eqn: _sym_and(*inexprs),
    "or": lambda inexprs, eqn: _sym_or(*inexprs),
    "max": lambda inexprs, eqn: _sym_max(*inexprs),
    "min": lambda inexprs, eqn: _sym_min(*inexprs),
    "logaddexp": lambda inexprs, eqn: _sym_logaddexp(*inexprs),
    "broadcast_in_dim": lambda inexprs, eqn: _sym_broadcast_in_dim(
        inexprs,
        shape=eqn.params["shape"],
        broadcast_dimensions=eqn.params["broadcast_dimensions"],
    ),
    "concatenate": lambda inexprs, eqn: _sym_concatenate(inexprs, eqn.params["dimension"]),
    "stack": lambda inexprs, eqn: _sym_stack(inexprs, eqn.params["axis"]),
    "dynamic_slice": lambda inexprs, eqn: _sym_dynamic_slice(inexprs, eqn),
    "slice": lambda inexprs, eqn: cast(Callable[..., ObjectArray], _sym_slice)(
        *inexprs,
        eqn.params["start_indices"],
        eqn.params["limit_indices"],
        eqn.params.get("strides", None),
    ),
    "squeeze": lambda inexprs, eqn: cast(Callable[..., ObjectArray], _sym_squeeze)(
        *inexprs, eqn.params["dimensions"]
    ),
    "scatter": lambda inexprs, eqn: cast(Callable[..., ObjectArray], _sym_scatter)(
        *inexprs,
        dimension_numbers=eqn.params["dimension_numbers"],
        mode=eqn.params["mode"],
    ),
    # 未验证（2026-07-14 加入，尚无数值回归覆盖）
    "scatter-add": lambda inexprs, eqn: cast(Callable[..., ObjectArray], _sym_scatter)(
        *inexprs,
        dimension_numbers=eqn.params["dimension_numbers"],
        mode=eqn.params["mode"],
        combine=lambda old, new: old + new,
    ),
    "gather": lambda inexprs, eqn: cast(Callable[..., ObjectArray], _sym_gather)(
        *inexprs,
        dimension_numbers=eqn.params["dimension_numbers"],
        slice_sizes=eqn.params["slice_sizes"],
        mode=eqn.params["mode"],
    ),
    "add_any": lambda inexprs, eqn: _sym_add_any(inexprs),
    "pad": lambda inexprs, eqn: _sym_pad(inexprs, eqn),
    "select_n": lambda inexprs, eqn: _sym_select_n(inexprs),
    "reduce_sum": lambda inexprs, eqn: _sym_reduce_sum(inexprs, eqn),
    "reduce_max": lambda inexprs, eqn: _sym_reduce_extreme(inexprs, eqn, sy.Max),
    "reduce_min": lambda inexprs, eqn: _sym_reduce_extreme(inexprs, eqn, sy.Min),
    "reduce_or": lambda inexprs, eqn: _sym_reduce_or(inexprs, eqn),
    # 未验证（2026-07-14 加入，尚无数值回归覆盖）
    "atan2": lambda inexprs, eqn: _sym_atan2(*inexprs),
    # 未验证（2026-07-14 加入，尚无数值回归覆盖）
    "cumsum": lambda inexprs, eqn: _sym_cumsum(inexprs, eqn),
    # 调试输出只有副作用、没有数学值，映射为空输出列表。
    "debug_print": lambda inexprs, eqn: [],
    "iota": lambda inexprs, eqn: _sym_iota(eqn),
    "split": lambda inexprs, eqn: _sym_split(inexprs, eqn),
    "convert_element_type": lambda inexprs, eqn: _sym_convert_element_type(inexprs[0]),
    "dot_general": lambda inexprs, eqn: _sym_dot_general(inexprs[0], inexprs[1], eqn),
    "transpose": lambda inexprs, eqn: _sym_transpose(inexprs[0], eqn),
    "tile": lambda inexprs, eqn: np.tile(inexprs[0], eqn.params["reps"]),
    "reshape": lambda inexprs, eqn: cast(Callable[..., ObjectArray], _sym_reshape)(
        *inexprs,  # 被 reshape 的表达式数组
        eqn.params["new_sizes"],
        eqn.params["dimensions"],
    ),
}


def get_sym_invar(
    invars: Sequence[core.Atom], var2sym: Mapping[core.Var, ObjectArray]
) -> list[ObjectArray]:
    """把方程输入转换成公式数组，不做隐式广播或常量折叠。

    字面量用 exact_array 精确转成 SymPy 常量数组；其余变量必须已在
    var2sym 中登记，缺失时由字典查找直接报错而不是静默忽略。
    """
    inexprs = []
    for inv in invars:
        if isinstance(inv, Literal):
            inexprs.append(exact_array(inv.val))
        else:
            inexprs.append(var2sym[inv])
    return inexprs


def get_sym_outvar(inexprs: Sequence[ObjectArray], eqn: core.JaxprEqn) -> list[SymbolicArray]:
    """按 primitive 名称查表得到输出公式；多输出原语返回 list。

    未支持的原语抛 NotImplementedError，不返回占位表达式；单输出结果统一
    包装成单元素 list，便于调用方按输出端口对齐。
    """
    prim_name = eqn.primitive.name
    if prim_name not in primitive_to_sympy_op:
        raise NotImplementedError(f"No Sympy translation rule for JAX primitive '{prim_name}'.")
    out_expr = primitive_to_sympy_op[prim_name](inexprs, eqn)
    if not isinstance(out_expr, list):  # 多变量输出由映射返回 list
        return [out_expr]
    else:
        return out_expr


def jaxpr_to_sympy_expressions(
    jaxpr: core.Jaxpr,
    var2sym: dict[core.Var, ObjectArray] | None = None,
    x_cnt: int = 0,
    c_cnt: int = 0,
) -> tuple[list[ObjectArray], dict[core.Var, ObjectArray], int, int]:
    """把 ClosedJaxpr 逐方程转换成 SymPy 表达式。

    先为函数输入分配带类型假设的符号数组、为捕获常量写入精确值，然后按
    方程顺序转换；jit/pjit 内部图递归转换后用符号替换把内联结果接回当前
    图。x_cnt/c_cnt 在嵌套调用间继续编号，使符号名全局唯一。
    返回（输出公式列表、变量映射、输入计数、常量计数）。
    不跟踪定义条件、有限性和预算；表外原语直接抛 NotImplementedError。
    """
    if var2sym is None:
        var2sym = {}

    def append_vars(
        vars: Sequence[core.Var], var2sym: dict[core.Var, ObjectArray], sym_name: str, cnt: int
    ) -> tuple[dict[core.Var, ObjectArray], int]:
        """为一组输入分配符号数组并登记到 var2sym，返回递增后的计数器。"""
        for i, var in enumerate(vars):  # 遍历 jaxpr 的全部输入
            assert var not in var2sym  # 确认没有覆盖已有绑定
            aval = cast(core.ShapedArray, var.aval)
            shape = aval.shape  # 输入的抽象形状
            num_elements = int(np.prod(shape))
            kind = np.dtype(aval.dtype).kind
            # 浮点输入取实数假设，整数输入取整数假设，布尔不加额外假设。
            assumptions = (
                {"real": True} if kind == "f" else {"integer": True} if kind in "iu" else {}
            )
            symbols_list = sy.symbols(
                [f"{sym_name}{i + cnt}_{j}" for j in range(num_elements)], **assumptions
            )
            symbolic_array = np.array(symbols_list).reshape(shape)
            var2sym[var] = symbolic_array
        return var2sym, cnt + len(vars)

    def substitute_symbol(symbol: sy.Basic, sym_map: Mapping[sy.Basic, sy.Basic]) -> sy.Basic:
        """把子图符号替换成当前图输入对应的表达式。"""
        return symbol.subs(sym_map)

    var2sym, x_cnt = append_vars(jaxpr.jaxpr.invars, var2sym, "x", cnt=x_cnt)
    # 捕获常量按精确值绑定，不引入新的自由符号身份。
    for var, value in zip(jaxpr.jaxpr.constvars, jaxpr.consts, strict=True):
        var2sym[var] = exact_array(value)
    c_cnt += len(jaxpr.jaxpr.constvars)

    for i, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive.name in ("pjit", "jit"):
            # 递归转换子图，再用符号映射把子图输入替换成当前图的表达式。
            _jaxpr = cast(core.Jaxpr, eqn.params.get("jaxpr", None))
            sub_syms_out, sub_var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(
                _jaxpr, var2sym={}, x_cnt=x_cnt, c_cnt=c_cnt
            )
            syms_in = get_sym_invar(eqn.invars, var2sym)
            # 子图符号按“先输入、后常量”的登记顺序排列，前 len(syms_in) 个
            # 是输入；常量已是精确值，不需要再代入。
            iterator = iter(sub_var2sym)
            sub_syms_in = [sub_var2sym[next(iterator)] for _ in range(len(syms_in))]
            sym_map = {}
            for sym_in, sub_sym_in in zip(syms_in, sub_syms_in):
                sym_map.update({key: value for key, value in zip(sub_sym_in.flat, sym_in.flat)})
            sub_syms_out = [np.vectorize(substitute_symbol)(f, sym_map) for f in sub_syms_out]
            exprs = sub_syms_out
        else:
            inexprs = get_sym_invar(eqn.invars, var2sym)
            exprs = get_sym_outvar(inexprs, eqn)
        exprs = [
            np.asarray(expr) for expr in exprs
        ]  # reshape/索引后不能让表达式退化成标量而丢失数组形状
        for outvar, expr in zip(eqn.outvars, exprs):
            assert outvar.aval.shape == expr.shape
            var2sym[outvar] = expr

    out_syms = get_sym_invar(jaxpr.jaxpr.outvars, var2sym)
    return out_syms, var2sym, x_cnt, c_cnt
