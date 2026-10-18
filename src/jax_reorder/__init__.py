"""CPU 上比较两份 JAX 计算图是否等价的公开接口。

按手里的输入选入口：
- 有 Python 函数：compare_functions，会自己追踪并返回 FunctionComparison；
- 有带常量绑定的 jaxpr：compare_algebraic_jaxprs；
- 只想看结构：compare_jaxprs 或 canonicalize；
- 想先追踪再比较：trace_function 加 compare_function_traces。

结论怎么读：函数入口默认只查前向，显式请求 jvp / vjp 才比较对应的一阶求导
程序。JVP 是雅可比乘输入切向量，VJP 是输出余切向量反向传播；FunctionComparison
按模式分别保存结果。图入口默认还比较 custom_jvp / custom_vjp 携带的一阶
规则，include_derivative_rules=False 可只看前向。

结构不同时，函数入口默认交给 SymPy 再做数学判定；equivalence="structural"
则只停留在结构层。inequivalent 表示有完整输出的数学不等价证据，different 只是
结构差异诊断，不推出数学结论。检查未完成、契约不兼容或运行失败都抛异常。

注意边界：代数结论建立在“理想实数”解释上，会附带浮点警告，不代表一般 Python
等价或逐位浮点等价。等价时 Comparison.correspondence 保存原图对应证据，
verify_correspondence 可以单独复核，不必重跑规范化。

实现分三层：core（规范化与结构比较）→ algebra（SymPy 数学判定）→ functions
（追踪与契约编排），errors 提供跨层共用的异常。子模块路径是实现细节，对外只以
本模块的 __all__ 为准。
"""

from .algebra import AlgebraComparison, compare_algebraic_jaxprs
from .core.canonical import Canonicalized, Comparison, canonicalize, compare_jaxprs
from .core.correspondence import EquationCorrespondence, GraphCorrespondence, verify_correspondence
from .errors import (
    ComparisonBudgetError,
    ComparisonContractError,
    ComparisonError,
    ComparisonFailureKind,
    ComparisonInternalError,
    ComparisonStage,
    ComparisonTimeoutError,
    CompilationError,
    CorrespondenceError,
    InvalidJaxprError,
    JaxprError,
    MathematicalUndecidedError,
    TraceError,
    UnsupportedJaxprError,
)
from .functions import (
    FunctionComparison,
    FunctionSignature,
    FunctionTrace,
    ModeComparison,
    compare_function_traces,
    compare_functions,
    trace_function,
)
from .status import Status

# 对外只导出结果类型、显式错误与调用入口，内部编码器和搜索状态留在各模块中。
__all__ = [
    "AlgebraComparison",  # 数学结论、假设、警告、结构对应或 SymPy 查询记录。
    "Canonicalized",  # 单图规范化结果：重排 jaxpr、比较结构、BFS/执行顺序、深度和变量映射。
    "Comparison",  # 双图比较报告：状态、差异或问题、两侧规范化结果以及原图对应证据。
    "EquationCorrespondence",  # 单个左侧方程对应的右侧方程索引、结果端口映射及子图证据。
    "FunctionComparison",  # 函数比较总报告：各模式结论、未检查模式和 CPU 编译检查结果。
    "FunctionSignature",  # 函数契约：输入输出 pytree、静态参数、可微叶子及变换顺序。
    "FunctionTrace",  # 单模式追踪产物：函数契约、环境、物化 jaxpr 和动态输入抽象签名。
    "GraphCorrespondence",  # 整张原图的对应证据：输入、常量、方程及输出的左到右双射。
    "ModeComparison",  # 单个 forward/JVP/VJP 模式的报告，保留图比较与追踪结果。
    "Status",  # equivalent、inequivalent；different 只用于显式结构诊断。
    "ComparisonError",
    "ComparisonStage",
    "ComparisonFailureKind",
    "ComparisonContractError",
    "ComparisonBudgetError",
    "ComparisonTimeoutError",
    "ComparisonInternalError",
    "MathematicalUndecidedError",
    "TraceError",
    "CompilationError",
    "InvalidJaxprError",  # 图不合法：重复定义、悬空引用、依赖环或 JAX 类型检查失败。
    "JaxprError",  # 图处理异常基类，携带 path 和 reason，便于统一捕获和定位问题。
    "CorrespondenceError",  # 对应证据缺失、映射不完整或绑定不保持语义时的错误。
    "UnsupportedJaxprError",  # 当前无法解释的语义、未审查版本等导致的错误。
    "canonicalize",  # 规范化一个 jaxpr，返回 Canonicalized；不支持或非法输入会抛出异常。
    "compare_algebraic_jaxprs",  # 结构检查后，用 SymPy 比较完整输出及定义条件。
    "compare_jaxprs",  # 规范化并比较两张图，再复核等价证据；图处理失败直接抛出异常。
    "compare_function_traces",  # 核对两份追踪的模式、环境、契约及图，返回 ModeComparison。
    "compare_functions",  # 按给定签名逐模式追踪并比较两个函数，返回 FunctionComparison。
    "trace_function",  # 按输入签名及指定 forward/JVP/VJP 模式追踪一个函数，返回 FunctionTrace。
    "verify_correspondence",  # 直接核对原图及给定映射；校验通过返回 None，失败抛出异常。
]

# 包自身版本；JAX/jaxlib 的受审查版本由规范化和对应校验入口另行检查。
__version__ = "0.2.0"
