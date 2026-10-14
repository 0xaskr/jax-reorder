"""求和域双射的微小检查内核，不搜索规范形式，也不识别 JAX 原语。

每个见证先验证适用条件，再改变绑定和坐标。平移区间以带符号前缀
表示，因而域覆盖、重叠与重数最终由精确多项式系数决定。
"""

from dataclasses import replace
from itertools import product

from .algebra_indices import add, coordinates, divide, integer, leaves, map_factors, scale
from .algebra_ir import FiniteRange, Index, Term, ratio


def ranges(term):
    """绑定域的标准半开表示。"""
    return {
        Index("bound", i): ((0, n) if type(n) is int else (n.start, n.stop))
        for i, n in enumerate(term.bounds)
    }


def offset_of(index, axis):
    """只接纳单个单位系数轴加非负常数，避免前缀扩展产生越界坐标。"""
    variable = Index("bound", axis)
    parts = index.args if index.kind == "add" else (index,)
    if sum(part == variable for part in parts) != 1:
        return None
    if any(part != variable and part.kind not in ("zero", "const") for part in parts):
        return None
    offset = sum(part.number for part in parts if part != variable)
    return offset if offset >= 0 else None


def translation(term, axis):
    """所有捕获该绑定的坐标必须有相同偏移。"""
    offsets = []
    variable = Index("bound", axis)

    def positive(index):
        # 已接入的视图坐标由非负偏移、正进制和正步长构成。
        # 负系数可能使前缀读取超过原上界，负偏移可能引入负坐标。
        return (index.kind not in ("const", "scale") or index.number >= 0) and all(
            positive(child) for child in index.args
        )

    def visit(index, period=0):
        if variable not in set(leaves(index)):
            return True
        offset = offset_of(index, axis)
        if offset is not None:
            offsets.append((offset, period))
            return True
        if index.kind == "mod":
            return visit(index.args[0], index.number)
        if index.kind == "div":
            return visit(index.args[0], period * index.number)
        return bool(index.args) and all(visit(child, 0) for child in index.args)

    for coordinate in coordinates(term.factors):
        if variable in set(leaves(coordinate)) and not positive(coordinate):
            return None
        if not visit(coordinate):
            return None
    rigid = {offset for offset, period in offsets if not period}
    if len(rigid) != 1:
        # 纯周期读取不需要域平移；避免循环移动同一周期。
        return None
    candidate = next(iter(rigid))
    if all(
        (candidate - offset) % period == 0 if period else candidate == offset
        for offset, period in offsets
    ):
        return candidate
    return None


def apply_domain_witness(term, witness):
    """检查单个换元见证；失败抛 ValueError，由证明器转为证据错误。"""
    kind, axis, amount = witness
    if type(axis) is not int or not 0 <= axis < len(term.bounds) or type(amount) is not int:
        raise ValueError("invalid domain witness axis")
    extent = term.bounds[axis]
    if type(extent) is not int or extent < 0:
        raise ValueError("domain witness requires a zero-based interval")
    variable = Index("bound", axis)
    if kind == "split":
        if not 1 < amount < extent or extent % amount:
            raise ValueError("domain split is not an integer bijection")
        new_bounds = list(term.bounds)
        new_bounds[axis] = extent // amount
        new_bounds.append(amount)
        mapped = add(scale(variable, amount), Index("bound", len(term.bounds)))

        def change(i):
            return mapped if i == variable else i
    elif kind == "fuse":
        if (
            not 0 <= amount < len(term.bounds)
            or amount == axis
            or type(term.bounds[amount]) is not int
            or min(extent, term.bounds[amount]) < 1
        ):
            raise ValueError("invalid domain fusion")
        minor = Index("bound", amount)
        radix = term.bounds[amount]
        new_bounds = list(term.bounds)
        new_bounds[axis] = extent * radix
        new_bounds[amount] = 1

        def change(i):
            if i == variable:
                return divide(variable, radix)
            if i == minor:
                return divide(variable, radix, modulo=True)
            return i
    elif kind == "shift":
        if (
            amount <= 0
            or not all(f.total for f in term.factors)
            or translation(term, axis) != amount
        ):
            raise ValueError("unsafe domain prefix extension")
        new_bounds = list(term.bounds)
        new_bounds[axis] = FiniteRange(amount, amount + extent)

        def change(i):
            return add(i, integer(-amount)) if i == variable else i
    else:
        raise ValueError("unknown domain witness")
    transformed = replace(term, bounds=tuple(new_bounds))
    return replace(transformed, factors=map_factors(term.factors, change, ranges(transformed)))


def prefix_terms(term, budget):
    """矩形域包含排除；先核对组合大小，再分配所有带符号前缀。"""
    choices = [((1, n),) if type(n) is int else ((1, n.stop), (-1, n.start)) for n in term.bounds]
    count = 1
    for choice in choices:
        count *= len(choice)
    budget.spend(count * (1 + len(term.factors) + len(term.bounds)), "algebra.domain_prefix")
    output = []
    for combination in product(*choices):
        sign, bounds = 1, []
        for polarity, stop in combination:
            sign *= polarity
            bounds.append(stop)
        output.append(Term(ratio(term.rational * sign), tuple(bounds), term.factors))
    return output
