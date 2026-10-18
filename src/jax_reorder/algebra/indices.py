"""符号张量布局的精确整数坐标适配。

Index 是不可变的整数表达式 DAG，只表示常量、自由/求和坐标叶子以及加法、
整数倍、整除和取余。所有化简都是整数恒等式，不需要额外前提，也不枚举
张量元素；维度长度只保留为范围。free/sum 叶子由调用方连同 domains 区间
一起提供，本模块不判断任意符号表达式是否语义等价，也不处理分数坐标。
"""

from dataclasses import dataclass

from ..core.unrooted import order_key


@dataclass(frozen=True)
class Index:
    """符号整数坐标；算术节点不枚举元素。

    kind 描述节点类别：zero 是唯一身份的 0，const 用 number 保存非零值，
    free/sum 是由外部构造的自由/绑定坐标叶子（free 的 number 是张量轴号，
    sum 的 number 是唯一的绑定编号），add 的 args 是已排序的加项，
    scale 的 number 为整数系数且 args 只有一项，div/mod 的 number 为按该
    除数做的整除/取余且 args 只有被除数。对象是冻结数据类，可直接作字典键；
    相等即结构相等，args 顺序有意义。
    """

    kind: str
    number: int = 0
    args: tuple["Index", ...] = ()


def integer(value):
    """零使用唯一身份，其他整数作为常量坐标。

    统一 zero 表示可以让 0 的相等判断与哈希不依赖构造来源；非零值保存
    在 const 节点的 number 上。
    """
    return Index("const", value) if value else Index("zero")


def scale(index, value):
    """对整数线性组合合并系数。

    0 与 zero 直接坍缩为 0；系数 1 返回原节点；常量与已有 scale 就地折叠，
    add 则把系数分配到各项（保持线性组合的规范形状）。其余情况保留一个
    scale 节点，等 add 收集系数时再合并；负数系数同样适用。
    """
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
    """规范化精确线性组合，并应用 div/mod 的重组恒等式。

    先展平嵌套 add、累加常量并逐个基节点收集整数系数（free/sum/div/mod
    等叶子按身份作基）。随后只在系数满足 q_weight == r_weight * divisor 时，
    用 q*divisor + r = base 把 mod 与其商一起还原成 base，避免遗漏前提。
    最后丢弃零系数，按 order_key 排序，使同一线性组合只有一种表示。
    """
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
    """保守闭区间推导；未知范围不作猜测。

    domains 把坐标叶子映射到半开区间 (start, stop)，这里换算成闭区间
    [start, max(start, stop-1)]；mod 的结果范围固定为 [0, number-1]。
    加法端点相加、整数倍按符号重排端点、div 对端点做向下取整。
    任一子项范围未知就返回 None，调用方据此放弃需要范围前提的化简。
    """
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
    """整除与取余；仅在已证明余项范围内消去混合进制坐标。

    与 Python 的 // 和 % 一致，向下取整，因此负坐标也满足
    index == (index // divisor) * divisor + index % divisor。
    已知恒等式先折叠：常量直接求值、除数为 1、嵌套 div 合并除数、
    内层模数是外层除数倍数时直接取余。对加法逐项拆分时，只有余项的闭区间
    落在 [0, divisor) 内才消去整除数部分；否则保留 div/mod 节点，等 bounds
    提供更多范围后再尝试。modulo 为 False 时返回商，为 True 时返回余数。
    divisor 必须是正整数，0 或负除数直接报错。
    """
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
    # 余项已在 [0, divisor) 内：商就是已分离的整数部分，余数就是 remainder。
    if interval is not None and 0 <= interval[0] <= interval[1] < divisor:
        return remainder if modulo else add(*divisible)
    # 取余时整除数部分对结果无贡献；对余项重新规范化仍可能需要相同规则。
    if modulo and any(i.kind != "zero" for i in divisible):
        return divide(remainder, divisor, domains, modulo=True)
    return Index("mod" if modulo else "div", divisor, (index,))
