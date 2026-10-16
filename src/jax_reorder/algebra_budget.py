"""Deterministic work budget shared by graph conversion and symbolic queries."""

from .errors import UnsupportedJaxprError


class Budget:
    """共享工作量预算；必须在分配项、组合或搜索候选之前扣减。"""

    def __init__(self, limit):
        """只接受正整数，布尔值不作为预算。"""
        if type(limit) is not int or limit < 1:
            raise ValueError("algebra_budget must be a positive integer")
        self.limit = limit
        self.used = 0

    def spend(self, amount=1, path="algebra"):
        """超限立即中止，不把未完成的转换或查询当作结论。"""
        if amount < 0 or amount > self.limit - self.used:
            raise UnsupportedJaxprError(
                path,
                f"algebra budget exceeded: used {self.used}, need {amount}, limit {self.limit}",
            )
        self.used += amount
