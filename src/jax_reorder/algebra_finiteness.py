"""有限性事实的 SSA 分析；不执行数组，也不展开张量元素。

特别验证 where(mask, softmax(where(mask, finite, -inf)), 0) 的最终值。
保留 softmax 内部的未定义全遮蔽路径；这里只证明最后选择的值有限，
原始定义条件继续由 SymPy 构造器保存。
"""

from dataclasses import dataclass

import jax
import numpy as np
from jax import lax
from jax._src import core


@dataclass(eq=False)
class Source:
    primitive: object
    shape: tuple
    dtype: object
    inputs: tuple = ()
    params: object = None
    finite: bool = False
    constant: object = None


def _coordinates(shape):
    return tuple(0 if size == 1 else ("axis", i) for i, size in enumerate(shape))


def _read(source, shape):
    """将逐点广播读取还原到原值及有序坐标，保留每一个非单位轴。"""
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
            indices = tuple(
                0 if size == 1 else indices[axis]
                for size, axis in zip(
                    child.shape, source.params["broadcast_dimensions"], strict=True
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


def _scalar(source, value):
    """只认经过视图/转换的实际标量常量，不从打印文本猜测常量。"""
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
            # Some floating formats have no infinity (for example e4m3fn).
            # Check every conversion: a later widening cannot recover an -inf
            # that an intermediate dtype converted into NaN.  This evaluates
            # only the scalar literal 0/-inf, never the traced tensor values.
            try:
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    if np.asarray(value, dtype=source.dtype).item() != value:
                        return False
            except (TypeError, ValueError, OverflowError):
                return False
        source = source.inputs[0]
    return source.primitive == "constant" and source.shape == () and source.constant == value


def _same_read(left, right, shape):
    a, b = _read(left, shape), _read(right, shape)
    return a is not None and a == b


def _reduction(source, operand, primitive):
    access = _read(source, operand.shape)
    if access is None:
        return None
    node, coordinates = access
    if node.primitive is not primitive or node.inputs[0] is not operand:
        return None
    axes = tuple(node.params["axes"])
    if not axes or any(operand.shape[axis] == 0 for axis in axes):
        return None
    expected = tuple(c for i, c in enumerate(_coordinates(operand.shape)) if i not in axes)
    return axes if coordinates == expected else None


def _guarded_softmax(source):
    if source.primitive is not lax.select_n_p or len(source.inputs) != 3:
        return False
    if not jax.dtypes.issubdtype(source.dtype, np.floating):
        return False
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
    # jax.nn.softmax includes maximum(-inf, reduce_max(...)) before keepdims.
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
        # Preserve the broadcast route through max rather than assuming that its
        # result has the expected free axes.
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
        axes = tuple(reduced.params["axes"])
        expected = tuple(c for i, c in enumerate(_coordinates(masked.shape)) if i not in axes)
        if not axes or any(masked.shape[axis] == 0 for axis in axes) or composed != expected:
            return False
    else:
        axes = _reduction(maximum, masked, lax.reduce_max_p)
        if axes is None:
            return False
    return _reduction(denominator, numerator, lax.reduce_sum_p) == axes


class Finiteness:
    """根据有序源输入与参数判定有限性，记录所用规则。"""

    def __init__(self):
        self.rules = set()

    def is_finite(self, source):
        op, args, params = source.primitive, source.inputs, source.params
        if _guarded_softmax(source):
            self.rules.add("guarded_softmax_finite")
            return True
        if op is lax.select_n_p:
            return all(a.finite for a in args[1:])
        if op in {lax.reduce_max_p, lax.reduce_min_p}:
            return args[0].finite and all(args[0].shape[i] > 0 for i in params["axes"])
        if op is lax.gather_p:
            # A defined gather reads operand elements or its declared fill.
            # This does not establish bounds/totality of the gather itself.
            default_nan = (
                params.get("mode") is lax.GatherScatterMode.FILL_OR_DROP
                and params.get("fill_value") is None
                and jax.dtypes.issubdtype(source.dtype, np.inexact)
            )
            return args[0].finite and not default_nan and not nonfinite_parameters(params)
        if op in {lax.dynamic_slice_p, lax.dynamic_update_slice_p}:
            return all(a.finite for a in args[: 2 if op is lax.dynamic_update_slice_p else 1])
        if op is lax.convert_element_type_p:
            same_type = source.dtype == args[0].dtype
            widening = all(
                jax.dtypes.issubdtype(dtype, np.floating) for dtype in (source.dtype, args[0].dtype)
            ) and float(jax.dtypes.finfo(source.dtype).max) >= float(
                jax.dtypes.finfo(args[0].dtype).max
            )
            return args[0].finite and (same_type or widening)
        if op is lax.tanh_p and not jax.dtypes.issubdtype(source.dtype, np.floating):
            # tanh has poles on the complex domain; only the real rule is total.
            return False
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


def nonfinite_parameters(value):
    """gather fill_value 等参数也能产生非有限值，不能只检查 SSA 常量。"""
    if isinstance(value, (float, complex, np.number, np.ndarray)):
        array = np.asarray(value)
        return jax.dtypes.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array))
    if isinstance(value, dict):
        return any(nonfinite_parameters(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(nonfinite_parameters(v) for v in value)
    return False


def nonfinite_graph(graph, budget, cache=None):
    """检查共同调用中的非有限常量与复数极点；循环反馈采用保守上界。"""
    if cache is None:
        cache = {}
    if graph in cache:
        return cache[graph]
    from .correspondence import _parameter_view

    def nonfinite(value, aval):
        return jax.dtypes.issubdtype(aval.dtype, np.inexact) and not np.all(
            np.isfinite(np.asarray(value, dtype=aval.dtype))
        )

    found = any(
        nonfinite(value, var.aval) for var, value in zip(graph.constvars, graph.consts, strict=True)
    )
    for eq in graph.eqns:
        budget.spend(path="algebra.nonfinite_graph")
        found |= eq.primitive is lax.tanh_p and any(
            jax.dtypes.issubdtype(var.aval.dtype, np.complexfloating) for var in eq.invars
        )
        found |= nonfinite_parameters(eq.params)
        found |= any(
            nonfinite(var.val, var.aval) for var in eq.invars if isinstance(var, core.Literal)
        )
        _, children = _parameter_view(eq, lambda _: None, derivative_scope="primal", parent=graph)
        for child in children.values():
            found |= nonfinite_graph(child, budget, cache)
    found |= any(
        nonfinite(var.val, var.aval) for var in graph.outvars if isinstance(var, core.Literal)
    )
    cache[graph] = bool(found)
    return bool(found)
