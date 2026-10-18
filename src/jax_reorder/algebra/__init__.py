"""代数层：结构对不上时，再用 SymPy 判断两边是不是数学上等价。

公共入口和结果类型由本模块导出，其余子模块是实现细节：
- compare：编排（结构快路径、契约检查、冻结、worker 查询）；
- budget：确定性的工作量预算；
- adapter：把 JAX 图桥接给数学层（元数据、默认 JIT、共同调用、跨进程冻结）；
- expression：符号值模型（坐标、常量、重命名）；
- symbolic：SymbolicBuilder 和逐原语映射；
- regions：cond/scan/while 区域的符号绑定；
- sympy_ops：受预算约束的化简与恒等查询；
- trusted：数学结果协议、独立 worker 和超时回收；
- explicit：保留的逐元素参考实现，不参与生产判等。
"""

from .compare import AlgebraComparison, compare_algebraic_jaxprs

__all__ = ["AlgebraComparison", "compare_algebraic_jaxprs"]
