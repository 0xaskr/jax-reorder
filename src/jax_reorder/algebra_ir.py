"""索引代数的不可变表示、精确系数及确定性资源预算。

自由轴按输出位置编号，求和临时轴仅在转换时存在；规范项的绑定轴局部编号。
标量调用捕获外部坐标，未知张量边界保持独立作用域；两者显式区分。
"""

from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction

from .errors import UnsupportedJaxprError


@dataclass(frozen=True)
class Index:
    """符号整数坐标；算术节点不枚举元素，scope 区分标量调用内外的绑定。"""

    kind: str
    number: int = 0
    args: tuple["Index", ...] = ()
    scope: int = 0


class _ImmutableDag:
    """缓存不可变节点哈希；精确比较显式遍历共享 DAG，避免指数重复访问。"""

    def __post_init__(self):
        """子节点的哈希已经缓存，构造父节点只处理一层。"""
        object.__setattr__(
            self,
            "_cached_hash",
            hash((type(self), tuple(getattr(self, f.name) for f in fields(self)))),
        )

    def __hash__(self):
        """哈希只用于查找；碰撞仍由完整内容比较处理。"""
        return self._cached_hash

    def __eq__(self, other):
        """按类型和字段精确比较，对共享子节点对进行去重。"""
        if type(self) is not type(other):
            return NotImplemented
        if hash(self) != hash(other):
            return False
        pending, seen = [(self, other)], set()
        while pending:
            left, right = pending.pop()
            if left is right:
                continue
            if type(left) is not type(right):
                return False
            pair = id(left), id(right)
            if pair in seen:
                continue
            seen.add(pair)
            if isinstance(left, _ImmutableDag) and hash(left) != hash(right):
                return False
            if is_dataclass(left) and not isinstance(left, type):
                pending.extend(
                    (getattr(left, f.name), getattr(right, f.name))
                    for f in fields(left)
                    if f.compare
                )
            elif isinstance(left, tuple):
                if len(left) != len(right):
                    return False
                pending.extend(zip(left, right, strict=True))
            elif left != right:
                return False
        return True


@dataclass(frozen=True, eq=False)
class Expr(_ImmutableDag):
    """逐元素语义；call 参数共享索引环境，atom 参数是独立张量边界。"""

    op: str
    data: tuple = ()
    args: tuple["Expr", ...] = ()
    total: bool = True


@dataclass(frozen=True)
class Recurrence:
    """有限递推的元素语义：状态、初值、更新及每步可观察输出均有序保存。"""

    time: Index
    length: int
    reverse: bool
    states: tuple[Expr, ...]
    initials: tuple[Expr, ...]
    updates: tuple[Expr, ...]
    emissions: tuple[Expr, ...]


@dataclass(frozen=True, eq=False)
class Factor(_ImmutableDag):
    """标量因子；scalar 参数中的 bound.scope 标识捕获哪一层的求和轴。"""

    semantic: tuple
    indices: tuple[Index, ...]
    arguments: tuple[tuple["Term", ...], ...] = ()
    total: bool = True
    scalar: bool = False


@dataclass(frozen=True, eq=False)
class FiniteRange(_ImmutableDag):
    """半开整数求和域；规范结果使用从零开始的域，平移阶段保留原端点。"""

    start: int
    stop: int


@dataclass(frozen=True, eq=False)
class Term(_ImmutableDag):
    """系数乘以有限求和中的因子乘积；系数保存为约分后的整数对。"""

    coefficient: tuple[int, int]
    bounds: tuple[int | FiniteRange, ...] = ()
    factors: tuple[Factor, ...] = ()

    @property
    def rational(self):
        """还原精确有理数，避免系数运算引入浮点舍入。"""
        return Fraction(*self.coefficient)


@dataclass(frozen=True)
class AlgebraStep:
    """一个表达式的局部推导，orders 提供每个展开项的合法索引置换。"""

    expression: Expr
    premises: tuple[int, ...]
    result: tuple[Term, ...]
    orders: tuple[tuple[int, ...], ...]
    domains: tuple[tuple[tuple, ...], ...] = ()


class Budget:
    """共享工作量预算；必须在分配项、组合或搜索候选之前扣减。"""

    def __init__(self, limit):
        """只接受正整数，布尔值不作为预算。"""
        if type(limit) is not int or limit < 1:
            raise ValueError("algebra_budget must be a positive integer")
        self.limit = limit
        self.used = 0

    def spend(self, amount=1, path="algebra"):
        """超限立即中止，不返回尚未完成搜索的局部规范形式。"""
        if amount < 0 or amount > self.limit - self.used:
            raise UnsupportedJaxprError(
                path,
                f"algebra budget exceeded: used {self.used}, need {amount}, limit {self.limit}",
            )
        self.used += amount

    def coefficient(self, value, path="algebra.coefficient"):
        """为大整数位数计费，限制精确计算的存储增长。"""
        self.spend(
            max(1, (value.numerator.bit_length() + value.denominator.bit_length()) // 64), path
        )


def ratio(value):
    """把精确运算结果保存为可编码和可排序的整数对。"""
    value = Fraction(value)
    return value.numerator, value.denominator
