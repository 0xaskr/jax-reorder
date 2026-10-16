"""Exact integer coordinate adaptation for symbolic tensor layouts."""

from dataclasses import dataclass

from .unrooted import order_key


@dataclass(frozen=True)
class Index:
    """符号整数坐标；算术节点不枚举元素。"""

    kind: str
    number: int = 0
    args: tuple["Index", ...] = ()


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
