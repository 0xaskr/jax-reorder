"""已审核逐点原语到标量函数的转换，与形状映射和求和域分离。"""

import jax
import numpy as np
from jax import lax

from .algebra_ir import Expr
from .algebra_lower import encode


def lower_scalar(lowerer, eq, values, params):
    """保留精确调用属性；参数共享当前输出元素的坐标。"""
    if eq.primitive is lax.convert_element_type_p:
        from .algebra_piecewise import _plain

        if (
            not _plain(eq)
            or len(values) != 1
            or set(params) != {"new_dtype", "weak_type", "sharding"}
            or params["sharding"] is not None
        ):
            return None
        aval = eq.outvars[0].aval
        if not all(
            jax.dtypes.issubdtype(dtype, np.floating) for dtype in (values[0].dtype, aval.dtype)
        ):
            return None
        body = lowerer._read(values[0], lowerer._axes(aval))
        if values[0].dtype == aval.dtype:
            # 只改变 weak_type 的转换不改变元素值；输出接口仍保存实际抽象值。
            return (lowerer._tensor(aval, body),)
        semantic = ("scalar_cast", encode(values[0].dtype), encode(params))
        lowerer.rules.add("scalar:convert_element_type")
        return (lowerer._tensor(aval, Expr("call", semantic, (body,), body.total)),)
    total_functions = {lax.sin_p, lax.cos_p, lax.tanh_p, lax.exp_p, lax.max_p, lax.min_p}
    partial_functions = {lax.log_p, lax.sqrt_p, lax.div_p}
    if eq.primitive not in total_functions | partial_functions:
        return None
    if params and params != {"accuracy": None}:
        return None
    if not lowerer._algebra_allowed(eq, values):
        return None
    aval = eq.outvars[0].aval
    indices = lowerer._axes(aval)
    arguments = tuple(lowerer._broadcast_read(value, aval.shape, indices) for value in values)
    semantic = ("scalar", eq.primitive, encode(params), encode(aval.dtype))
    total = eq.primitive in total_functions and all(a.total for a in arguments)
    lowerer.rules.add(f"scalar:{eq.primitive}")
    return (lowerer._tensor(aval, Expr("call", semantic, arguments, total)),)
