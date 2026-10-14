"""在完整矩形域内证实安全后，以前缀指标表达静态零扩展。"""

import numpy as np
from jax import lax

from .algebra_indices import add, bounds, integer, map_factors
from .algebra_ir import Expr, Index
from .unrooted import order_key


def simplify_prefix_factors(factors, *, key=order_key):
    """指标 I(0<=i<1) 为 1 时 i 必为 0；保留指标自身以保留支持域。"""
    if any(not factor.total for factor in factors):
        return factors

    def prefix(factor):
        return (
            len(factor.semantic) == 2
            and factor.semantic[0] == "index_prefix"
            and type(factor.semantic[1]) is int
            and factor.semantic[1] > 0
            and len(factor.indices) == 1
            and not factor.arguments
            and not factor.scalar
        )

    singletons = {
        factor.indices[0]
        for factor in factors
        if prefix(factor)
        and factor.semantic[1] == 1
        and factor.indices[0].kind in {"free", "sum", "bound"}
        and factor.indices[0].scope == 0
    }
    result, anchors = [], set()
    for factor in factors:
        if prefix(factor) and factor.semantic[1] == 1 and factor.indices[0] in singletons:
            if factor not in anchors:
                result.append(factor)
                anchors.add(factor)
            continue
        if singletons:
            (factor,) = map_factors(
                (factor,), lambda index: Index("zero") if index in singletons else index, key=key
            )
        if prefix(factor) and factor.indices[0].kind in {"zero", "const"}:
            if 0 <= factor.indices[0].number < factor.semantic[1]:
                continue
        result.append(factor)
    return tuple(result)


def _extends_safely(lowerer, expression, shape):
    """扩展可能读取原切片之外；每个原始读取都必须仍在实际张量范围内。"""
    domains = {**lowerer.index_domains, **{Index("free", i): (0, n) for i, n in enumerate(shape)}}
    pending, seen = [expression], set()
    while pending:
        body = pending.pop()
        if body in seen:
            continue
        seen.add(body)
        lowerer.budget.spend(path="algebra.padding_domain")
        if not body.total:
            return False
        if body.op == "atom":
            semantic, indices = body.data
            if semantic[:1] == ("index_prefix",):
                continue
            source_shape = lowerer.read_shapes.get(semantic)
            if source_shape is None or len(source_shape) != len(indices):
                return False
            for index, extent in zip(indices, source_shape, strict=True):
                interval = bounds(index, domains)
                if interval is None or not 0 <= interval[0] <= interval[1] < extent:
                    return False
            # 张量调用的实参属于自己的坐标环境；其求值 barrier 独立保留。
        elif body.op in {"constant", "add", "mul", "sum", "call"}:
            pending.extend(body.args)
        else:
            return False
    return True


def lower_padding(lowerer, eq, values, params):
    """只接受静态边缘零填充；没有安全扩展证明则保留原调用。"""
    if (
        eq.primitive is not lax.pad_p
        or set(params) != {"padding_config"}
        or len(values) != 2
        or not lowerer._algebra_allowed(eq, values)
        or values[1].shape
        or values[1].body != Expr("constant", (0, 1))
    ):
        return None
    source, _ = values
    padding = params["padding_config"]
    if len(padding) != len(source.shape) or any(
        len(part) != 3
        or any(not isinstance(n, (int, np.integer)) or isinstance(n, bool) for n in part)
        or part[0] < 0
        or part[1] < 0
        or part[2] != 0
        for part in padding
    ):
        return None
    aval = eq.outvars[0].aval
    indices = lowerer._axes(aval)
    body = lowerer._read(
        source,
        tuple(add(index, integer(-int(p[0]))) for index, p in zip(indices, padding, strict=True)),
    )
    if 0 in source.shape and source.body.total:
        body = Expr("constant", (0, 1))
    elif not _extends_safely(lowerer, body, aval.shape):
        return None
    else:

        def prefix(index, stop, extent):
            if stop <= 0:
                return Expr("constant", (0, 1))
            if stop >= extent:
                return Expr("constant", (1, 1))
            return Expr("atom", (("index_prefix", stop), (index,)))

        for index, size, extent, (low, _, _) in zip(
            indices, source.shape, aval.shape, padding, strict=True
        ):
            low = int(low)
            mask = Expr(
                "add",
                args=(
                    prefix(index, low + size, extent),
                    Expr("mul", args=(Expr("constant", (-1, 1)), prefix(index, low, extent))),
                ),
            )
            body = Expr("mul", args=(body, mask))
    lowerer.rules.add("safe_zero_extension")
    return (lowerer._tensor(aval, body),)
