"""整个包的比较结论词汇。

四种对外结果都带一个 status 字段，含义由承载它的结果类型决定：
- Comparison（结构比较）：EQUIVALENT / DIFFERENT；
- AlgebraComparison（数学比较）：EQUIVALENT / INEQUIVALENT；
- ModeComparison / FunctionComparison：三种都可能，按最保守规则汇总。

DIFFERENT 只是结构诊断，不表示数学不等价；失败不是状态，而是异常。
结构等价比数学等价更强：结构等价 ⇒ 数学等价，但数学等价 ⇏ 结构等价。
"""

from enum import StrEnum


class Status(StrEnum):
    """比较结论；DIFFERENT 只用于结构诊断，失败通过异常报告。"""

    # 字符串值可直接写入结构化报告。
    EQUIVALENT = "equivalent"
    INEQUIVALENT = "inequivalent"
    DIFFERENT = "different"
