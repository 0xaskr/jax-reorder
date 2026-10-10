"""CPU JAX 计算图规范化与比较的公开接口。

canonicalize 返回重排图及结构信息；compare_jaxprs 返回图比较报告。
结果仅描述给定 jaxpr 的结构语义，不代表一般 Python 或代数等价性证明。
"""

from .canonical import Canonicalized, Comparison, Status, canonicalize, compare_jaxprs
from .errors import InvalidJaxprError, JaxprError, UnsupportedJaxprError

__all__ = [
    "Canonicalized",
    "Comparison",
    "InvalidJaxprError",
    "JaxprError",
    "Status",
    "UnsupportedJaxprError",
    "canonicalize",
    "compare_jaxprs",
]

__version__ = "0.1.0"
