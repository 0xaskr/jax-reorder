"""非法图与不支持语义的显式错误，避免通过丢弃信息获得匹配。

canonicalize 向调用方抛出这些错误；compare_jaxprs 捕获后生成带位置和原因的
unknown 报告。区分输入不合法与实现尚未支持，方便排查调用问题或补充适配。
"""


class JaxprError(ValueError):
    """图处理错误的基类，提供可直接用于比较报告的位置与原因。"""

    def __init__(self, path: str, reason: str):
        # 分开保存，比较器才能给位置追加 left/right 前缀，而不必解析异常字符串。
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class InvalidJaxprError(JaxprError):
    """输入违反 jaxpr 表示或类型规则，例如重复定义、悬空引用或依赖环。"""


class UnsupportedJaxprError(JaxprError):
    """输入包含当前尚不能完整解释的语义，不意味着 JAX 自身一定不支持。"""


class CorrespondenceError(JaxprError):
    """原始图对应证据不完整或不保持语义，不能据此报告等价。"""
