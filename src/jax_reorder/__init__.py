"""CPU JAX 计算图规范化与比较的公开接口。

canonicalize 返回重排图及结构信息；compare_jaxprs 返回图比较报告。
结果仅描述给定 jaxpr 的结构语义，不代表一般 Python 或代数等价性证明。
"""

from .canonical import Canonicalized, Comparison, Status, canonicalize, compare_jaxprs
from .correspondence import EquationCorrespondence, GraphCorrespondence, verify_correspondence
from .errors import CorrespondenceError, InvalidJaxprError, JaxprError, UnsupportedJaxprError
from .functions import (
    FunctionComparison,
    FunctionSignature,
    FunctionTrace,
    ModeComparison,
    compare_function_traces,
    compare_functions,
    trace_function,
)

__all__ = [
    "Canonicalized",
    "Comparison",
    "CorrespondenceError",
    "EquationCorrespondence",
    "FunctionComparison",
    "FunctionSignature",
    "FunctionTrace",
    "GraphCorrespondence",
    "InvalidJaxprError",
    "JaxprError",
    "ModeComparison",
    "Status",
    "UnsupportedJaxprError",
    "canonicalize",
    "compare_function_traces",
    "compare_functions",
    "compare_jaxprs",
    "trace_function",
    "verify_correspondence",
]

__version__ = "0.1.0"
