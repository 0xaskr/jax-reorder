"""将已证明为独立累加的 scan 转为有限求和，不枚举循环长度。

只消去每个反馈量均满足 ``c' = c + delta(xs, consts)`` 的纯实数循环。
中间序列必须与变化的反馈量无关；其余循环仍交给原有区域比较。
"""

import jax
import numpy as np
from jax import lax
from jax._src import core
from jax._src.layout import AutoLayout

from .algebra_indices import leaves, map_index
from .algebra_ir import Expr, Recurrence


def _ordinary(aval):
    """循环消去不得抹去特殊布局、分片、内存位置或非实数类型。"""
    return (
        isinstance(aval, core.ShapedArray)
        and all(type(n) is int and n >= 0 for n in aval.shape)
        and jax.dtypes.issubdtype(aval.dtype, np.floating)
        and aval.memory_space is core.MemorySpace.Device
        and aval.layout is AutoLayout
        and aval.manual_axis_type.empty
        and aval.sharding.mesh.empty
    )


def _mentions(expression, semantics, budget):
    """检查全部子表达式，包括算子的独立张量参数中隐藏的反馈引用。"""
    pending, seen = [expression], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        budget.spend(path="algebra.scan.dependencies")
        seen.add(current)
        if current.op == "atom" and current.data[0] in semantics:
            return True
        pending.extend(current.args)
    return False


def _increment(expression, own, all_semantics, budget):
    """只接受恰好一次、未变换的自身反馈，其余加项不得引用任一反馈。"""
    pending, independent, occurrences = [expression], [], 0
    while pending:
        current = pending.pop()
        budget.spend(path="algebra.scan.feedback")
        if current == own:
            occurrences += 1
        elif current.op == "add":
            pending.extend(current.args)
        elif _mentions(current, all_semantics, budget):
            return None
        else:
            independent.append(current)
    if occurrences != 1:
        return None
    return Expr("add", args=tuple(independent)) if independent else Expr("constant", (0, 1))


def _captures_time(expression, time, budget):
    """独立 atom 参数不能捕获外部求和索引，避免跨张量作用域误绑定。"""
    pending, seen = [(expression, False)], set()
    while pending:
        current, inside_argument = pending.pop()
        key = current, inside_argument
        if key in seen:
            continue
        budget.spend(path="algebra.scan.scope")
        seen.add(key)
        if current.op in ("atom", "bag"):
            if inside_argument and any(time in set(leaves(i)) for i in current.data[1]):
                return True
            pending.extend((child, True) for child in current.args)
        else:
            pending.extend((child, inside_argument) for child in current.args)
    return False


def _replace(expression, replacements, time, time_value, lowerer, cache):
    """代入恒定反馈与序列输出的时间轴，独立张量参数保留自身自由轴。"""
    if expression in cache:
        return cache[expression]
    lowerer.budget.spend(path="algebra.scan.substitute")
    if expression.op in ("atom", "bag"):
        semantic, indices = expression.data
        indices = tuple(map_index(i, lambda j: time_value if j == time else j) for i in indices)
        if semantic in replacements:
            result = lowerer._read(replacements[semantic], indices)
        else:
            result = Expr(
                expression.op,
                (semantic, indices),
                tuple(
                    _replace(child, replacements, time, time_value, lowerer, cache)
                    for child in expression.args
                ),
                expression.total,
            )
    else:
        result = Expr(
            expression.op,
            expression.data,
            tuple(
                _replace(child, replacements, time, time_value, lowerer, cache)
                for child in expression.args
            ),
            expression.total,
        )
    cache[expression] = result
    return result


def _solve_accumulators(lowerer, recurrence, initials):
    """在递推 IR 上证明独立累加，不依赖 scan 参数或原语组合。"""
    # 反向求导经常把恒定的 cotangent 作为 carry 传递；先证明其不变，
    # 才能将其初值代入其他累加器的增量，不能把变化的 carry 当成常量。
    fixed = {
        own.data[0]: initial
        for out, own, initial in zip(recurrence.updates, recurrence.states, initials, strict=True)
        if out == own
    }
    semantics = {state.data[0] for state in recurrence.states}
    increments = tuple(
        Expr("constant", (0, 1))
        if own.data[0] in fixed
        else _increment(
            _replace(out, fixed, recurrence.time, recurrence.time, lowerer, {}),
            own,
            semantics,
            lowerer.budget,
        )
        for out, own in zip(recurrence.updates, recurrence.states, strict=True)
    )
    if any(value is None for value in increments):
        return None
    if any(
        _captures_time(v, recurrence.time, lowerer.budget)
        for v in (*recurrence.updates, *recurrence.emissions)
    ):
        return None
    changing = set()
    for initial, own, increment in zip(initials, recurrence.states, increments, strict=True):
        if increment == Expr("constant", (0, 1)):
            fixed[own.data[0]] = initial
        else:
            changing.add(own.data[0])
    if any(_mentions(v, changing, lowerer.budget) for v in recurrence.emissions):
        return None

    return increments, fixed


def lower_scan(lowerer, eq, values, params):
    """返回全部 carry 和 ys；不能证明该适配完整有效时返回 None。"""
    # 局部导入让公共 lowerer 可以直接调用本适配器。
    from .algebra_lower import encode

    if eq.primitive is not lax.scan_p or set(params) != {
        "jaxpr",
        "ft_in",
        "ft_out",
        "length",
        "reverse",
        "unroll",
    }:
        return None
    length = params["length"]
    if (
        type(length) is not int
        or length < 0
        or type(params["reverse"]) is not bool
        or type(params["unroll"]) is not int
        or params["unroll"] < 0
        or eq.effects
        or eq.ctx.compute_type is not None
        or not eq.ctx.cur_abstract_mesh.empty
        or eq.ctx.xla_metadata is not None
        or eq.ctx.cur_layout_mode is not core.LayoutMode.AUTO
        or not all(_ordinary(v.aval) for v in (*eq.invars, *eq.outvars))
        or not all(v.body.total for v in values)
    ):
        return None
    ft_in, ft_out, body = params["ft_in"], params["ft_out"], params["jaxpr"]
    if (
        not ft_in.unpackable
        or not ft_out.unpackable
        or any(v is not None for v in (*ft_in.vals, *ft_out.vals))
    ):
        return None
    input_groups, output_groups = ft_in.unpack(), ft_out.unpack()
    if len(input_groups) != 3 or len(output_groups) != 2:
        return None
    nc, nk, nx = map(len, input_groups)
    no, ny = map(len, output_groups)
    if (
        nk != no
        or len(values) != nc + nk + nx
        or len(body.invars) != len(values)
        or len(body.outvars) != nk + ny
        or len(eq.outvars) != nk + ny
        or not all(_ordinary(v.aval) for v in (*body.invars, *body.outvars))
    ):
        return None
    for i, (outer, inner) in enumerate(zip(eq.invars, body.invars, strict=True)):
        shape = outer.aval.shape
        if i >= nc + nk:
            if not shape or shape[0] != length:
                return None
            shape = shape[1:]
        if encode(inner.aval) != encode(outer.aval.update(shape=shape)):
            return None
    for i, (outer, inner) in enumerate(zip(eq.outvars, body.outvars, strict=True)):
        shape = inner.aval.shape if i < nk else (length, *inner.aval.shape)
        if encode(outer.aval) != encode(inner.aval.update(shape=shape)):
            return None
        if i < nk and encode(inner.aval) != encode(
            body.invars[nc + i].aval.update(weak_type=inner.aval.weak_type)
        ):
            return None

    time = lowerer._new_index()
    lowerer.index_domains[time] = (0, length)
    carry = tuple(
        lowerer._input(v.aval, ("scan_accumulate", time.number, i))
        for i, v in enumerate(body.invars[nc : nc + nk])
    )
    slices = tuple(
        lowerer._tensor(var.aval, lowerer._read(value, (time, *lowerer._axes(var.aval))))
        for var, value in zip(body.invars[nc + nk :], values[nc + nk :], strict=True)
    )
    lowered = lowerer.graph(body, (*values[:nc], *carry, *slices))
    if lowered.barriers or any(not v.body.total for v in lowered.outputs):
        return None
    recurrence = Recurrence(
        time,
        length,
        params["reverse"],
        tuple(v.body for v in carry),
        tuple(v.body for v in values[nc : nc + nk]),
        tuple(v.body for v in lowered.outputs[:nk]),
        tuple(v.body for v in lowered.outputs[nk:]),
    )
    solved = _solve_accumulators(lowerer, recurrence, values[nc : nc + nk])
    if solved is None:
        return None
    increments, fixed = solved

    outputs = []
    for i, (initial, increment) in enumerate(zip(recurrence.initials, increments, strict=True)):
        reduced = lowerer._sum(increment, ((recurrence.time, recurrence.length),))
        expression = Expr("add", args=(initial, reduced))
        outputs.append(lowerer._tensor(eq.outvars[i].aval, expression))
    for i, value in enumerate(lowered.outputs[nk:], nk):
        axes = lowerer._axes(eq.outvars[i].aval)
        shifted = lowerer._read(value, axes[1:])
        expression = _replace(shifted, fixed, time, axes[0], lowerer, {})
        outputs.append(lowerer._tensor(eq.outvars[i].aval, expression))
    lowerer.rules.add("scan:independent_accumulation")
    return tuple(outputs)
