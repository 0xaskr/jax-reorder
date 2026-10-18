"""用 SSA 分析证明某些子表达式一定是有限值，不执行数组，也不展开张量元素。

典型例子是 where(mask, softmax(where(mask, finite, -inf)), 0)：softmax 内部
全遮蔽的路径仍然没有被证明有定义，这里只能证明最终选出来的值是有限的。
有限性只是辅助事实，原始定义条件仍由 SymPy 构造器保存，不能拿它替代。
分析以 SSA 值（Source 对象身份）为单位，只有显式列出的规则能给出 True，
其余一律保守地当作未知，不做一般的 IEEE 浮点推理。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import numpy as np
from jax import lax
from jax._src import core
from jax.typing import DTypeLike

from ..core._typing import Parameters
from ._typing import Shape
from .budget import Budget

type ReadCoordinate = int | tuple[Literal["axis"], int]


@dataclass(eq=False)
class Source:
    """一个 SSA 值的有限性摘要；eq=False 使对象身份即 SSA 定义。

    primitive/shape/dtype/inputs/params 记录产生该值的操作；constant 只对
    标量常量节点保存实际值。finite 表示“在已声明前提和保留的定义条件下可作
    有限实数解释”，不表示该值一定已定义，也不表示任意 dtype 都有限。
    两个结构相同的值在这里仍是不同定义，避免把不同读取误认为同一来源。
    """

    primitive: core.Primitive | str | None
    shape: Shape
    dtype: DTypeLike
    inputs: tuple[Source, ...] = ()
    params: Parameters | None = None
    finite: bool = False
    constant: object = None


def _coordinates(shape: Sequence[int]) -> tuple[ReadCoordinate, ...]:
    """返回 shape 各轴的坐标符号；单位轴固定为 0。

    单位轴没有可区分取值，用 0 表示后，广播读取只需比较其余轴，避免因
    多写一个恒为 0 的轴而错判不同读取。
    """
    return tuple(0 if size == 1 else ("axis", i) for i, size in enumerate(shape))


def _read(source: Source, shape: Shape) -> tuple[Source, tuple[ReadCoordinate, ...]] | None:
    """将逐点广播读取还原到原值及有序坐标，保留每一个非单位轴。

    shape 是更大（或相同）的读取坐标域；source 必须能按末轴后缀广播到
    shape，否则返回 None。随后穿过 broadcast_in_dim、stop_gradient 和
    同 dtype 的 convert_element_type 继续上溯，直到遇到不透明操作数。
    返回 (基值 Source, 基值 shape 上的坐标元组)；坐标中的单位轴为 0。
    """
    if len(source.shape) > len(shape):
        return None
    suffix = shape[len(shape) - len(source.shape) :] if source.shape else ()
    if any(a not in (1, b) for a, b in zip(source.shape, suffix, strict=True)):
        return None
    coordinates = _coordinates(shape)
    indices = tuple(
        0 if size == 1 else coordinates[len(shape) - len(source.shape) + i]
        for i, size in enumerate(source.shape)
    )
    while source.inputs:
        if source.primitive is lax.broadcast_in_dim_p:
            child = source.inputs[0]
            # 把当前 shape 的坐标按 broadcast_dimensions 投回子值 shape，
            # 子值的单位轴同样固定读 0。
            indices = tuple(
                0 if size == 1 else indices[axis]
                for size, axis in zip(
                    child.shape,
                    cast(Parameters, source.params)["broadcast_dimensions"],
                    strict=True,
                )
            )
        elif source.primitive is lax.stop_gradient_p:
            child = source.inputs[0]
        elif (
            source.primitive is lax.convert_element_type_p
            and source.dtype == source.inputs[0].dtype
        ):
            child = source.inputs[0]
        else:
            break
        source = child
    return source, indices


def _scalar(source: Source, value: float) -> bool:
    """只认经过视图/转换的实际标量常量，不从打印文本猜测常量。

    依次剥离 broadcast_in_dim、convert_element_type 和 stop_gradient；遇到
    浮点转换时额外核对转换是否保持数值：某些浮点格式没有无穷（如 e4m3fn），
    必须逐层检查，因为后续加宽无法恢复中间 dtype 已变成 NaN 的 -inf。
    这里只对字面量 0/-inf 做标量转换，不读取任何被追踪的张量数值。
    """
    while source.primitive in {
        lax.broadcast_in_dim_p,
        lax.convert_element_type_p,
        lax.stop_gradient_p,
    }:
        if source.primitive is lax.convert_element_type_p:
            if not (
                jax.dtypes.issubdtype(source.dtype, np.floating)
                and jax.dtypes.issubdtype(source.inputs[0].dtype, np.number)
            ):
                return False
            # 逐层检查：中间 dtype 把 -inf 变成 NaN 后，外层加宽也恢复不了原值。
            try:
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    if np.asarray(value, dtype=source.dtype).item() != value:
                        return False
            except (TypeError, ValueError, OverflowError):
                return False
        source = source.inputs[0]
    # 常量节点用字符串标签表示，不依赖 JAX primitive；必须是标量且内容相等。
    return source.primitive == "constant" and source.shape == () and source.constant == value


def _same_read(left: Source, right: Source, shape: Shape) -> bool:
    """判断两个值在同一 shape 上是否为同一次读取。

    Source 按对象身份比较，因此这要求两侧是同一个 SSA 定义，而不只是
    结构相同的表达式；广播路径和索引也必须完全一致。
    """
    a, b = _read(left, shape), _read(right, shape)
    return a is not None and a == b


def _reduction(
    source: Source, operand: Source, primitive: core.Primitive
) -> tuple[int, ...] | None:
    """识别对 operand 的完整归约读取，返回其轴元组或 None。

    要求 source 经视图/广播后读到 operand 的全部非归约坐标（没有置换或
    偏移），归约节点直接作用于 operand 本身，且被归约轴长度都大于 0：
    空轴归约产生单位元（例如 max 的 -inf），不能据此断言有限。
    """
    access = _read(source, operand.shape)
    if access is None:
        return None
    node, coordinates = access
    if node.primitive is not primitive or node.inputs[0] is not operand:
        return None
    axes = tuple(cast(Parameters, node.params)["axes"])
    if not axes or any(operand.shape[axis] == 0 for axis in axes):
        return None
    expected = tuple(c for i, c in enumerate(_coordinates(operand.shape)) if i not in axes)
    return axes if coordinates == expected else None


def _guarded_softmax(source: Source) -> bool:
    """识别 where(mask, softmax(where(mask, logits, -inf)), 0) 的有限性规则。

    必须同时核对：选择分支为浮点、假分支是标量 0、真分支是 exp(x-max)/sum
    形状的 softmax；mask 被同一个 SSA 值用于外层选择和 softmax 内部遮蔽，
    内层填 -inf，logits 本身已有限；max 与 sum 的归约轴完全一致。
    任一环节不同都不成立，尤其不能把“全遮蔽行靠最终 guard 定义为零”的
    结论推广到缺少同一 guard 的图。
    """
    if source.primitive is not lax.select_n_p or len(source.inputs) != 3:
        return False
    if not jax.dtypes.issubdtype(source.dtype, np.floating):
        return False
    # select_n_p 的操作数顺序是（pred，假分支，真分支），与 lax.select 相反。
    mask, otherwise, probabilities = source.inputs
    if mask.dtype != np.dtype("bool") or not _scalar(otherwise, 0):
        return False
    probability_access = _read(probabilities, source.shape)
    if probability_access is None:
        return False
    probabilities, coordinates = probability_access
    if (
        probabilities.primitive is not lax.div_p
        or probabilities.shape != source.shape
        or coordinates != _coordinates(source.shape)
    ):
        return False
    numerator, denominator = probabilities.inputs
    if numerator.primitive is not lax.exp_p or numerator.shape != source.shape:
        return False
    difference = numerator.inputs[0]
    if difference.primitive is not lax.sub_p or difference.shape != source.shape:
        return False
    masked, maximum = difference.inputs
    if masked.primitive is not lax.select_n_p or masked.shape != source.shape:
        return False
    inner_mask, padding, logits = masked.inputs
    if (
        not _scalar(padding, -np.inf)
        or not logits.finite
        or not _same_read(mask, inner_mask, source.shape)
    ):
        return False
    # jax.nn.softmax 在 keepdims 之前先做 maximum(-inf, reduce_max(...))，
    # 需要还原这条广播路径才能把 max 的轴与分母 sum 的轴对齐。
    maximum_access = _read(maximum, source.shape)
    if maximum_access is None:
        return False
    maximum_base, _ = maximum_access
    if maximum_base.primitive is lax.max_p:
        left, right = maximum_base.inputs
        if _scalar(left, -np.inf):
            reduced_max = right
        elif _scalar(right, -np.inf):
            reduced_max = left
        else:
            return False
        # 保留经过 max 的广播路径再比较坐标，不能假设其结果已经具有期望的自由轴。
        reduced_access = _read(reduced_max, maximum_base.shape)
        if reduced_access is None:
            return False
        reduced, inner_coordinates = reduced_access
        outer_coordinates = maximum_access[1]
        composed = tuple(
            outer_coordinates[c[1]] if isinstance(c, tuple) else c for c in inner_coordinates
        )
        if reduced.primitive is not lax.reduce_max_p or reduced.inputs[0] is not masked:
            return False
        axes = tuple(cast(Parameters, reduced.params)["axes"])
        expected = tuple(c for i, c in enumerate(_coordinates(masked.shape)) if i not in axes)
        if not axes or any(masked.shape[axis] == 0 for axis in axes) or composed != expected:
            return False
    else:
        axes = _reduction(maximum, masked, lax.reduce_max_p)
        if axes is None:
            return False
    return _reduction(denominator, numerator, lax.reduce_sum_p) == axes


class Finiteness:
    """根据有序源输入与参数判定有限性，记录所用规则。

    每个 Source 只是一条方程的局部摘要，其输入 Source 的 finite 已由同一
    SSA 遍历自底向上算出；本类不做递归、不重新解析图，也不执行任何数组。
    rules 记录实际触发的规则名，随转换结果一起报告，便于审计结论来源。
    """

    def __init__(self) -> None:
        """创建一个空的规则记录器；同一 builder 内复用可累计已触发的规则。"""
        self.rules: set[str] = set()

    def is_finite(self, source: Source) -> bool:
        """对一条已构造的 Source 应用白名单规则；未命中即保守返回 False。

        规则按特异性排列：先尝试 guarded-softmax 的完整模式，再处理逐点选择、
        归约极值、gather、动态切片、类型转换和实值初等函数。输出为整数或
        布尔时由调用方另行标记有限，不属于这里的实数有限性判断。
        """
        op, args, params = source.primitive, source.inputs, cast(Parameters, source.params)
        if _guarded_softmax(source):
            self.rules.add("guarded_softmax_finite")
            return True
        if op is lax.select_n_p:
            return all(a.finite for a in args[1:])
        if op in {lax.reduce_max_p, lax.reduce_min_p}:
            return args[0].finite and all(args[0].shape[i] > 0 for i in params["axes"])
        if op is lax.gather_p:
            # 已定义的 gather 只读取 operand 元素或声明的 fill；这不证明 gather
            # 本身的越界/全定义性质，只说明取值来源有限。浮点默认 fill 是 NaN，
            # 因此 FILL_OR_DROP 且未显式指定 fill 时拒绝，参数里的非有限值也拒绝。
            default_nan = (
                params.get("mode") is lax.GatherScatterMode.FILL_OR_DROP
                and params.get("fill_value") is None
                and jax.dtypes.issubdtype(source.dtype, np.inexact)
            )
            return args[0].finite and not default_nan and not nonfinite_parameters(params)
        if op in {lax.dynamic_slice_p, lax.dynamic_update_slice_p}:
            return all(a.finite for a in args[: 2 if op is lax.dynamic_update_slice_p else 1])
        if op is lax.convert_element_type_p:
            # 同 dtype 或浮点到更宽浮点的转换不会溢出成 inf；缩窄可能溢出，
            # 即使源值有限也不能断言结果有限。
            same_type = source.dtype == args[0].dtype
            widening = all(
                jax.dtypes.issubdtype(dtype, np.floating) for dtype in (source.dtype, args[0].dtype)
            ) and float(jax.dtypes.finfo(source.dtype).max) >= float(
                jax.dtypes.finfo(args[0].dtype).max
            )
            return args[0].finite and (same_type or widening)
        if op is lax.tanh_p and not jax.dtypes.issubdtype(source.dtype, np.floating):
            # tanh 在复平面有极点；只有实数规则是全定义的。
            return False
        # 下面这些逐点/视图/归约原语在操作数有限时保持有限：加乘、点积、
        # 非空轴求和、实值 exp/sin/cos/tanh/abs、max/min 与形状变换。
        safe = {
            lax.add_p,
            lax.sub_p,
            lax.mul_p,
            lax.neg_p,
            lax.dot_general_p,
            lax.reduce_sum_p,
            lax.exp_p,
            lax.sin_p,
            lax.cos_p,
            lax.tanh_p,
            lax.abs_p,
            lax.max_p,
            lax.min_p,
            lax.broadcast_in_dim_p,
            lax.reshape_p,
            lax.transpose_p,
            lax.squeeze_p,
            lax.slice_p,
            lax.concatenate_p,
            lax.stack_p,
            lax.stop_gradient_p,
        }
        return op in safe and all(a.finite for a in args)


def nonfinite_parameters(value: object) -> bool:
    """gather fill_value 等参数也能产生非有限值，不能只检查 SSA 常量。

    递归遍历 tuple/list/dict，只对 Python/NumPy 的浮点与复数容器检查；
    非浮点参数按原语义交给调用方，不在这里猜测。
    """
    if isinstance(value, (float, complex, np.number, np.ndarray)):
        array = np.asarray(value)
        return jax.dtypes.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array))
    if isinstance(value, dict):
        return any(nonfinite_parameters(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(nonfinite_parameters(v) for v in value)
    return False


def nonfinite_graph(
    graph: core.Jaxpr, budget: Budget, cache: dict[core.Jaxpr, bool] | None = None
) -> bool:
    """检查共同调用中的非有限常量与复数极点，任一处发现即保守返回。

    进入独立 worker 前调用：共同调用只有在整张子图不含非有限值时才允许
    共享符号，否则不共享、按普通 JIT 展开。常量数组按声明 dtype 转换后检查，
    参数里的非有限值、每个方程的字面量以及嵌套子图都要检查；嵌套区域
    （含循环体）逐层递归，任一处非有限就使整个调用不可共享。
    缓存以子图对象为键，共享区域只扫描一次；预算在每个方程上扣减，
    避免深层图无偿消耗。返回 True 表示“存在可能非有限的值”，不是有限性证明。
    """
    if cache is None:
        cache = {}
    if graph in cache:
        return cache[graph]
    from ..core.params import parameter_view

    def nonfinite(value: Any, aval: core.AbstractValue) -> bool:
        """按抽象值 dtype 转换后检查常量是否含非有限值；整数/布尔恒为有限。"""
        aval = cast(core.ShapedArray, aval)
        return jax.dtypes.issubdtype(aval.dtype, np.inexact) and not np.all(
            np.isfinite(np.asarray(value, dtype=aval.dtype))
        )

    found = any(
        nonfinite(value, var.aval) for var, value in zip(graph.constvars, graph.consts, strict=True)
    )
    for eq in graph.eqns:
        budget.spend(path="algebra.nonfinite_graph")
        # 复数 tanh 有极点；实数 tanh 是全定义的，不在此列。
        found |= eq.primitive is lax.tanh_p and any(
            jax.dtypes.issubdtype(cast(core.ShapedArray, var.aval).dtype, np.complexfloating)
            for var in eq.invars
        )
        found |= nonfinite_parameters(eq.params)
        found |= any(
            nonfinite(var.val, var.aval) for var in eq.invars if isinstance(var, core.Literal)
        )
        _, children = parameter_view(
            eq, lambda _: None, include_derivative_rules=False, parent=graph
        )
        for child in children.values():
            found |= nonfinite_graph(child, budget, cache)
    found |= any(
        nonfinite(var.val, var.aval) for var in graph.outvars if isinstance(var, core.Literal)
    )
    cache[graph] = bool(found)
    return bool(found)
