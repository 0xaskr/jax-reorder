"""把可逐点证明的实数分段选择转换为统一绝对值表达式。

比较条件只记录严格/非严格半轴，只有两条分支在零点相等时才消去
这个区别。普通选择仍保留条件，因此不会把 JAX 在零点不同的导数合并。
"""

import jax
import numpy as np
from jax import lax
from jax._src import core, pjit
from jax._src.api import Inline
from jax._src.layout import AutoLayout
from jax._src.sharding_impls import UNSPECIFIED

from .algebra_ir import Expr, Index
from .algebra_lower import encode


def _plain(eq):
    """仅允许默认执行属性；不能在数学适配中丢失布局或放置约束。"""
    if (
        eq.effects
        or eq.ctx.compute_type is not None
        or not eq.ctx.cur_abstract_mesh.empty
        or eq.ctx.xla_metadata is not None
        or eq.ctx.cur_layout_mode is not core.LayoutMode.AUTO
    ):
        return False
    for var in (*eq.invars, *eq.outvars):
        aval = var.aval
        if (
            not isinstance(aval, core.ShapedArray)
            or any(type(size) is not int or size < 0 for size in aval.shape)
            or aval.memory_space is not core.MemorySpace.Device
            or aval.layout is not AutoLayout
            or not aval.manual_axis_type.empty
            or not aval.sharding.mesh.empty
        ):
            return False
    return True


def _absolute(lowerer, aval, body):
    """绝对值只保留一个数学身份，参数仍是完整且有序的张量表达式。"""
    if 0 in aval.shape:
        # 空域没有任何输出元素，逐点绝对值与其输入相同。
        return lowerer._tensor(aval, body)
    return lowerer._tensor(
        aval,
        Expr(
            "call",
            ("real_absolute", encode(aval.dtype)),
            (body,),
            body.total,
        ),
    )


def _comparison(lowerer, eq, values):
    """记录 x 与精确零的比较；反向书写先恢复为 x 在左边。"""
    comparisons = {
        lax.gt_p: "gt",
        lax.ge_p: "ge",
        lax.lt_p: "lt",
        lax.le_p: "le",
        lax.eq_p: "eq",
        lax.ne_p: "ne",
    }
    if eq.primitive not in comparisons or eq.params or len(values) != 2:
        return None
    aval = eq.outvars[0].aval
    if (
        aval.dtype != np.dtype("bool")
        or values[0].dtype != values[1].dtype
        or not jax.dtypes.issubdtype(values[0].dtype, np.floating)
    ):
        return None
    relation = comparisons[eq.primitive]
    zero = Expr("constant", (0, 1))
    if values[1].body == zero:
        value = values[0]
    elif values[0].body == zero:
        value = values[1]
        relation = {"gt": "lt", "ge": "le", "lt": "gt", "le": "ge", "eq": "eq", "ne": "ne"}[
            relation
        ]
    else:
        args = tuple(lowerer._broadcast_read(v, aval.shape, lowerer._axes(aval)) for v in values)
        return (
            lowerer._tensor(
                aval,
                Expr(
                    "call",
                    ("real_comparison", relation, encode(values[0].dtype)),
                    args,
                    all(a.total for a in args),
                ),
            ),
        )
    source = lowerer._broadcast_read(value, aval.shape, lowerer._axes(aval))
    body = Expr("call", ("real_zero_comparison", relation), (source,), source.total)
    lowerer.rules.add("real_zero_comparison")
    return (lowerer._tensor(aval, body),)


def _select_absolute(lowerer, eq, values):
    """证明正半轴取 x、负半轴取 -x；零点两者都等于实数零。"""
    if eq.primitive is not lax.select_n_p or eq.params or len(values) != 3:
        return None
    if any(value.nonfinite for value in values):
        return None
    condition, false_value, true_value = values
    aval = eq.outvars[0].aval
    if condition.dtype != np.dtype("bool") or not lowerer._algebra_allowed(
        eq, (false_value, true_value)
    ):
        return None
    indices = lowerer._axes(aval)
    predicate = lowerer._broadcast_read(condition, aval.shape, indices)
    if predicate.op != "call" or predicate.data[0] != "real_zero_comparison":
        return None
    _, relation = predicate.data
    if relation not in ("gt", "ge", "lt", "le"):
        return None
    source = predicate.args[0]
    negative = Expr("mul", args=(Expr("constant", (-1, 1)), source), total=source.total)
    false_body = lowerer._broadcast_read(false_value, aval.shape, indices)
    true_body = lowerer._broadcast_read(true_value, aval.shape, indices)
    expected = (negative, source) if relation in ("gt", "ge") else (source, negative)
    if (false_body, true_body) != expected:
        return None
    lowerer.rules.add("select_absolute")
    return (_absolute(lowerer, aval, source),)


def _generic_select(lowerer, eq, values):
    """未满足绝对值恒等式的条件选择仍是带谓词的逐点标量函数。"""
    if eq.primitive is not lax.select_n_p or eq.params or len(values) != 3:
        return None
    if values[0].dtype != np.dtype("bool") or not lowerer._algebra_allowed(eq, values[1:]):
        return None
    aval = eq.outvars[0].aval
    args = tuple(lowerer._broadcast_read(v, aval.shape, lowerer._axes(aval)) for v in values)
    return (
        lowerer._tensor(
            aval,
            Expr("call", ("select", encode(aval.dtype)), args, all(a.total for a in args)),
        ),
    )


def _broadcast_condition(lowerer, eq, values):
    """布尔广播只改变读取坐标，不使用任何实数代数定律。"""
    if (
        eq.primitive is not lax.broadcast_in_dim_p
        or len(values) != 1
        or values[0].dtype != np.dtype("bool")
        or set(eq.params) - {"shape", "broadcast_dimensions", "sharding"}
        or eq.params.get("sharding") is not None
    ):
        return None
    aval = eq.outvars[0].aval
    indices = lowerer._axes(aval)
    body = lowerer._read(
        values[0],
        tuple(
            Index("zero") if size == 1 and aval.shape[dimension] != 1 else indices[dimension]
            for size, dimension in zip(
                values[0].shape, eq.params["broadcast_dimensions"], strict=True
            )
        ),
    )
    lowerer.rules.add("broadcast_condition")
    return (lowerer._tensor(aval, body),)


def _select_wrapper(lowerer, eq, values, params):
    """只展开逐点 select/广播包装；诊断名称不参与准入判断。"""
    required = {
        "jaxpr",
        "name",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "donated_invars",
        "ctx_mesh",
        "keep_unused",
        "inline",
        "compiler_options_kvs",
    }
    if eq.primitive is not pjit.jit_p or set(params) != required:
        return None
    if (
        any(item is not UNSPECIFIED for item in (*params["in_shardings"], *params["out_shardings"]))
        or any(item is not None for item in (*params["in_layouts"], *params["out_layouts"]))
        or any(item is not False for item in params["donated_invars"])
        or not params["ctx_mesh"].empty
        or params["keep_unused"] is not False
        or params["inline"] is not Inline.AUTO
        or params["compiler_options_kvs"] != ()
    ):
        return None
    child = params["jaxpr"]
    if child.effects or child.constvars or child.is_high:
        return None
    empty_identity = not child.eqns and all(0 in var.aval.shape for var in child.outvars)
    if not empty_identity and not any(node.primitive is lax.select_n_p for node in child.eqns):
        return None
    for node in child.eqns:
        if not _plain(node):
            return None
        if node.primitive is lax.select_n_p:
            if (
                node.params
                or len(node.invars) != 3
                or node.invars[0].aval.dtype != np.dtype("bool")
            ):
                return None
        elif node.primitive is lax.broadcast_in_dim_p:
            if (
                set(node.params) - {"shape", "broadcast_dimensions", "sharding"}
                or node.params.get("sharding") is not None
            ):
                return None
        else:
            return None
    if tuple(encode(var.aval) for var in child.invars) != tuple(
        value.signature for value in values
    ) or tuple(encode(var.aval) for var in child.outvars) != tuple(
        encode(var.aval) for var in eq.outvars
    ):
        return None
    used = set(var for var in child.outvars if isinstance(var, core.Var))
    for node in reversed(child.eqns):
        if any(var in used for var in node.outvars):
            used.update(var for var in node.invars if isinstance(var, core.Var))
    if any(
        var not in used and not value.body.total
        for var, value in zip(child.invars, values, strict=True)
    ):
        # 调用方已经计算的部分函数，不能因被包装函数没有读取它而消失。
        return None
    lowered = lowerer.graph(child, values)
    if lowered.barriers:
        return None
    lowerer.rules.add("transparent_select_wrapper")
    return lowered.outputs


def lower_piecewise(lowerer, eq, values, params):
    """在普通算术之后、区域边界之前尝试已审查的分段实数规则。"""
    if not _plain(eq):
        return None
    if (
        eq.primitive is lax.abs_p
        and not params
        and len(values) == 1
        and not values[0].nonfinite
        and lowerer._algebra_allowed(eq, values)
    ):
        aval = eq.outvars[0].aval
        lowerer.rules.add("real_absolute")
        return (_absolute(lowerer, aval, lowerer._read(values[0], lowerer._axes(aval))),)
    for adapter in (_comparison, _select_absolute, _generic_select, _broadcast_condition):
        result = adapter(lowerer, eq, values)
        if result is not None:
            return result
    return _select_wrapper(lowerer, eq, values, params)
