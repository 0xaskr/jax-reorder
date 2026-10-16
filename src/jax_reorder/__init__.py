"""CPU JAX 计算图规范化与比较的公开接口。

常用入口按已有输入选择：
- 手中是 Python 函数：compare_functions，返回 FunctionComparison。
- 手中是带常量绑定的 jaxpr：compare_jaxprs，返回 Comparison。
- 只需规范化一张图：canonicalize，返回 Canonicalized。
- 需要分开追踪和比较：先 trace_function，再 compare_function_traces。

函数入口默认只检查 forward；显式请求 jvp/vjp 才检查相应的一阶求导程序。
JVP 是雅可比与输入切向量的乘积，VJP 是输出余切向量经雅可比反向传播的结果。
图入口默认还比较已适配的自定义一阶规则，derivative_scope="primal" 限定为前向。

FunctionComparison.modes 保存各模式的 ModeComparison；其中的 comparison
在进入图比较后保存 Comparison。Comparison.left/right 是两侧的 Canonicalized，
correspondence 则是等价时生成并复核的 GraphCorrespondence 对应证据。
verify_correspondence 可独立复核这份证据，不重新执行规范化。

函数入口默认在结构不同时交给可信 SymPy 数学后端；equivalence="structural" 保留结构语义。
代数结论使用带明确浮点警告的理想实数解释，不代表一般 Python 或逐位浮点等价。
inequivalent 表示有完整输出的数学不等价证据；different 仅表示图或函数契约有差异；
unknown 表示检查未完成。
"""

from .algebra import AlgebraComparison, compare_algebraic_jaxprs
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
    "ModeComparison",  # 单个 forward/JVP/VJP 模式的报告，保留图比较、失败类别和追踪结果。
    "Status",  # equivalent、inequivalent、different（结构/契约）与 unknown。
    "InvalidJaxprError",  # 图不合法：重复定义、悬空引用、依赖环或 JAX 类型检查失败。
    "JaxprError",  # 图处理异常基类，携带 path 和 reason，便于统一捕获和定位问题。
    "CorrespondenceError",  # 对应证据缺失、映射不完整或绑定不保持语义时的错误。
    "UnsupportedJaxprError",  # 当前无法解释的语义、未审查版本或搜索预算不足等导致的错误。
    "canonicalize",  # 规范化一个 jaxpr，返回 Canonicalized；不支持或非法输入会抛出异常。
    "compare_algebraic_jaxprs",  # 结构检查后，用 SymPy 比较完整输出及定义条件。
    "compare_jaxprs",  # 规范化并比较两张图，再复核等价证据；图处理失败记录为 unknown。
    "compare_function_traces",  # 核对两份追踪的模式、环境、契约及图，返回 ModeComparison。
    "compare_functions",  # 按给定签名逐模式追踪并比较两个函数，返回 FunctionComparison。
    "trace_function",  # 按输入签名及指定 forward/JVP/VJP 模式追踪一个函数，返回 FunctionTrace。
    "verify_correspondence",  # 直接核对原图及给定映射；校验通过返回 None，失败抛出异常。
]

# 包自身版本；JAX/jaxlib 的受审查版本由规范化和对应校验入口另行检查。
__version__ = "0.1.0"
