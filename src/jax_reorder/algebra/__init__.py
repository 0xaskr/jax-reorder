"""代数层：结构校验后用可信 SymPy 判定数学等价。

公共入口与结果类型由本模块导出；其余子模块是实现细节，不承诺稳定导入路径：
- compare：结构快路径、契约检查、冻结与 worker 查询的编排。
- budget：确定性工作量预算与时间参数校验。
- adapter：JAX 图到数学层的桥接（元数据、默认 JIT、共同调用、进程边界冻结）。
- expression：符号值模型（SymbolicTensor、坐标与绑定重命名）。
- symbolic：SymbolicBuilder 与逐原语映射。
- regions：cond/scan/while 区域的符号绑定。
- sympy_ops：预算受控的化简与恒等查询。
- trusted：数学结果协议、独立 worker 与超时回收。
- explicit：保留的逐元素参考实现。
"""

from .compare import AlgebraComparison, compare_algebraic_jaxprs

__all__ = ["AlgebraComparison", "compare_algebraic_jaxprs"]
