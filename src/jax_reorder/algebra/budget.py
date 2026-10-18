"""转换图和查询 SymPy 时共用的工作量预算。

预算就是一个整数计数器：编码一个方程、构造一个索引、展开一项、执行一次
查询之前先扣减。同一份输入走同一条路径时总会在同一处停下，所以超预算可以
报告成确定的失败，而不是随调度变化的截断。它不衡量墙钟、内存或线程数；
墙钟上限由 worker 超时单独负责。
"""

from __future__ import annotations

from ..errors import ComparisonBudgetError, ComparisonStage


class Budget:
    """共享工作量预算；必须在分配项、组合或搜索候选之前扣减。

    每个代数比较流程把同一个 Budget 实例传遍转换和查询各层，使不同阶段
    的消耗累加到同一上限。预算不足时抛出异常，既不返回部分结果，也不把
    未完成的转换当作数学结论。
    """

    def __init__(self, limit: int) -> None:
        """只接受正整数，布尔值不作为预算。

        type 精确比较而非 isinstance：True 是 int 的子类，不能被当作 1。
        limit 为 0 或负数没有意义，直接拒绝而不是在首次 spend 时才报错。
        """
        if type(limit) is not int or limit < 1:
            raise ValueError("algebra_budget must be a positive integer")
        self.limit = limit
        self.used = 0

    def spend(self, amount: int = 1, path: str = "algebra") -> None:
        """超限立即中止，不把未完成的转换或查询当作结论。

        amount 必须非负；负扣减会凭空增加剩余预算。检查 amount 是否超过
        余额而不是先加后比较，可避免大 amount 造成中间值溢出。
        path 写入异常诊断，用于指出是哪一步先耗尽预算。
        """
        if amount < 0 or amount > self.limit - self.used:
            raise ComparisonBudgetError(
                path,
                f"algebra budget exceeded: used {self.used}, need {amount}, limit {self.limit}",
                stage=ComparisonStage.ALGEBRA,
                used=self.used,
                requested=amount,
                limit=self.limit,
            )
        self.used += amount


def validate_timeout(value: float) -> None:
    """要求正整数或浮点数的有限正超时；布尔值被显式排除。

    用 type 精确比较而不是 isinstance，避免 True/False 被当成 int；
    无穷与 NaN 也不能表达 worker 的墙钟上限。
    """
    if type(value) not in (int, float) or not 0 < value < float("inf"):
        raise ValueError("algebra_timeout must be a positive finite number")
