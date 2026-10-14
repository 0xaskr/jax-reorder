"""符号整数坐标及作用域操作；这里的规则只涉及精确整数，不读取数组。

free 是输出坐标，sum 是转换阶段的绑定身份，bound 是规范项内的局部轴。
标量函数参数共享外部坐标，但其中每个多项式的求和轴有自己的作用域。
"""

from dataclasses import replace

from .algebra_ir import Index
from .unrooted import order_key


def integer(value):
    """零使用唯一身份，其他整数作为常量坐标。"""
    return Index("const", value) if value else Index("zero")


def scale(index, value):
    """对整数线性组合合并系数。"""
    if not value or index.kind == "zero":
        return integer(0)
    if value == 1:
        return index
    if index.kind == "const":
        return integer(index.number * value)
    if index.kind == "scale":
        return scale(index.args[0], index.number * value)
    if index.kind == "add":
        return add(*(scale(child, value) for child in index.args))
    return Index("scale", value, (index,))


def add(*indices):
    """规范化精确线性组合，并应用 div/mod 的重组恒等式。"""
    pending, coefficients, constant = list(indices), {}, 0
    while pending:
        index = pending.pop()
        if index.kind == "add":
            pending.extend(index.args)
        elif index.kind in ("const", "zero"):
            constant += index.number
        else:
            weight, base = (index.number, index.args[0]) if index.kind == "scale" else (1, index)
            coefficients[base] = coefficients.get(base, 0) + weight
    # q*d+r 恢复原坐标；适用于任何整数，包含负值。
    for index in tuple(coefficients):
        if index.kind != "mod":
            continue
        weight = coefficients[index]
        quotient = Index("div", index.number, index.args)
        qweight = coefficients.get(quotient, 0)
        if weight and qweight == weight * index.number:
            coefficients[index] = coefficients[quotient] = 0
            base = index.args[0]
            coefficients[base] = coefficients.get(base, 0) + weight
    parts = [scale(base, weight) for base, weight in coefficients.items() if weight]
    if constant:
        parts.append(integer(constant))
    parts.sort(key=order_key)
    if not parts:
        return integer(0)
    if len(parts) == 1:
        return parts[0]
    return Index("add", args=tuple(parts))


def bounds(index, domains):
    """保守闭区间推导；未知范围不作猜测。"""
    if index.kind in ("zero", "const"):
        return index.number, index.number
    if index in domains:
        start, stop = domains[index]
        return start, max(start, stop - 1)
    if index.kind == "mod":
        return 0, index.number - 1
    if not index.args:
        return None
    ranges = [bounds(child, domains) for child in index.args]
    if any(item is None for item in ranges):
        return None
    if index.kind == "add":
        return sum(item[0] for item in ranges), sum(item[1] for item in ranges)
    if index.kind == "scale":
        a, b = (n * index.number for n in ranges[0])
        return min(a, b), max(a, b)
    if index.kind == "div":
        return tuple(n // index.number for n in ranges[0])
    return None


def divide(index, divisor, domains=None, *, modulo=False):
    """整除与取余；仅在已证明余项范围内消去混合进制坐标。"""
    if type(divisor) is not int or divisor < 1:
        raise ValueError("index divisor must be a positive integer")
    domains = {} if domains is None else domains
    if index.kind in ("zero", "const"):
        return integer(index.number % divisor if modulo else index.number // divisor)
    if divisor == 1:
        return integer(0) if modulo else index
    if not modulo and index.kind == "div":
        return divide(index.args[0], index.number * divisor, domains)
    if modulo and index.kind == "mod" and index.number % divisor == 0:
        return divide(index.args[0], divisor, domains, modulo=True)
    pieces = index.args if index.kind == "add" else (index,)
    divisible, rest = [], []
    for piece in pieces:
        if piece.kind == "scale" and piece.number % divisor == 0:
            divisible.append(scale(piece.args[0], piece.number // divisor))
        elif piece.kind in ("const", "zero"):
            divisible.append(integer(piece.number // divisor))
            rest.append(integer(piece.number % divisor))
        else:
            rest.append(piece)
    remainder = add(*rest)
    interval = bounds(remainder, domains)
    if interval is not None and 0 <= interval[0] <= interval[1] < divisor:
        return remainder if modulo else add(*divisible)
    if modulo and any(i.kind != "zero" for i in divisible):
        return divide(remainder, divisor, domains, modulo=True)
    return Index("mod" if modulo else "div", divisor, (index,))


def map_index(index, change, domains=None):
    """递归替换坐标叶子后，再以已知范围化简整数表达式。"""
    if not index.args:
        return change(index)
    args = tuple(map_index(child, change, domains) for child in index.args)
    if index.kind == "add":
        return add(*args)
    if index.kind == "scale":
        return scale(args[0], index.number)
    if index.kind in ("div", "mod"):
        return divide(args[0], index.number, domains, modulo=index.kind == "mod")
    raise ValueError(f"unknown index operation {index.kind}")


def leaves(index):
    """遍历一个坐标中的变量和常量。"""
    if not index.args:
        yield index
    for child in index.args:
        yield from leaves(child)


def map_factors(factors, change, domains=None, depth=0):
    """改写当前项可见的坐标；标量参数递归进入，张量边界参数保持独立。"""
    domains = {} if domains is None else domains
    scoped_domains = {
        replace(index, scope=index.scope + depth) if index.kind == "bound" else index: value
        for index, value in domains.items()
    }

    def local(index):
        if index.kind == "bound" and index.scope < depth:
            return index
        exposed = replace(index, scope=index.scope - depth) if index.kind == "bound" else index
        result = change(exposed)
        return map_index(
            result, lambda i: replace(i, scope=i.scope + depth) if i.kind == "bound" else i
        )

    output = []
    for factor in factors:
        indices = tuple(map_index(i, local, scoped_domains) for i in factor.indices)
        arguments = factor.arguments
        if factor.scalar:
            arguments = tuple(
                tuple(
                    sorted(
                        (
                            replace(
                                term,
                                factors=tuple(
                                    sorted(
                                        map_factors(term.factors, change, domains, depth + 1),
                                        key=order_key,
                                    )
                                ),
                            )
                            for term in polynomial
                        ),
                        key=order_key,
                    )
                )
                for polynomial in arguments
            )
        output.append(replace(factor, indices=indices, arguments=arguments))
    return tuple(output)


def coordinates(factors, depth=0):
    """列出当前项作用域中的读取坐标；返回值屏蔽参数内部的局部绑定。"""
    for factor in factors:
        for index in factor.indices:
            yield map_index(
                index, lambda i: replace(i, scope=i.scope - depth) if i.kind == "bound" else i
            )
        if factor.scalar:
            for polynomial in factor.arguments:
                for term in polynomial:
                    yield from coordinates(term.factors, depth + 1)


def used_axes(factors):
    """包括标量函数内部引用的外部求和轴。"""
    return {
        i.number
        for coordinate in coordinates(factors)
        for i in leaves(coordinate)
        if i.kind == "bound" and i.scope == 0
    }


def scalar_factor(semantic, children, total):
    """嵌入参数多项式时，已有外部绑定上移一级，局部绑定仍在原作用域。"""
    from .algebra_ir import Factor

    # children 的 bound 均由各自 Term 绑定，外部 sum 尚未转为 bound。
    return Factor(semantic, (), children, total, True)
