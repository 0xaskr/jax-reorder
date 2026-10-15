"""有限性事实的 SSA 分析；不执行数组，也不展开张量元素。

特别验证 where(mask, softmax(where(mask, finite, -inf)), 0) 的最终值。
保留 softmax 内部的未定义全遮蔽路径；这里只证明最后选择的值有限，
不把该事实提升为可以删除部分求值的 Expr.total。
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
    """每个分析实例给源输入独立身份，共同子表达式按精确参数共享。"""

    def __init__(self, lowerer):
        self.lowerer = lowerer
        self.interned = {}
        self.calls = {}

    def _constant(self, value, aval):
        from .algebra_lower import encode

        array = np.asarray(value, dtype=aval.dtype)
        key = "constant", encode(aval), encode(array)
        if key not in self.interned:
            self.interned[key] = Source(
                "constant",
                tuple(aval.shape),
                aval.dtype,
                finite=bool(np.all(np.isfinite(array))),
                constant=array.item() if array.shape == () else None,
            )
        return self.interned[key]

    def _finite(self, source):
        op, args, params = source.primitive, source.inputs, source.params
        if _guarded_softmax(source):
            self.lowerer.rules.add("guarded_softmax_finite")
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
            return (
                args[0].finite
                and not default_nan
                and not self.lowerer._nonfinite_parameters(params)
            )
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

    def graph(self, graph, bindings):
        from .algebra_jit import default_jit
        from .algebra_lower import encode
        from .algebra_piecewise import _plain

        environment = dict(zip(graph.invars, bindings, strict=True))
        environment.update(
            (var, self._constant(value, var.aval))
            for var, value in zip(graph.constvars, graph.consts, strict=True)
        )

        def read(var):
            return (
                self._constant(var.val, var.aval)
                if isinstance(var, core.Literal)
                else environment[var]
            )

        for eq in graph.eqns:
            self.lowerer.budget.spend(path="algebra.finiteness")
            for var in eq.outvars:
                self.lowerer._axes(var.aval)
            args = tuple(read(var) for var in eq.invars)
            if default_jit(eq, eq.params):
                child = eq.params["jaxpr"]
                key = child, args
                if key not in self.calls:
                    _, self.calls[key] = self.graph(child, args)
                outputs = self.calls[key]
            elif (
                _plain(eq)
                and len(eq.outvars) == 1
                and not any(
                    isinstance(value, (core.Jaxpr, core.ClosedJaxpr))
                    for value in eq.params.values()
                )
            ):
                aval = eq.outvars[0].aval
                try:
                    key = eq.primitive, encode(eq.params), encode(aval), args
                except (TypeError, ValueError):
                    key = object()
                if key not in self.interned:
                    node = Source(eq.primitive, tuple(aval.shape), aval.dtype, args, eq.params)
                    node.finite = self._finite(node)
                    self.interned[key] = node
                outputs = (self.interned[key],)
            else:
                outputs = tuple(Source(None, tuple(v.aval.shape), v.aval.dtype) for v in eq.outvars)
            environment.update(zip(eq.outvars, outputs, strict=True))
        return environment, tuple(read(var) for var in graph.outvars)


def analyze(lowerer, graph, bindings):
    inputs = tuple(
        Source(None, value.shape, value.dtype, finite=lowerer.finite_values.get(value, False))
        for value in bindings
    )
    return Finiteness(lowerer).graph(graph, inputs)[0]
