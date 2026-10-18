"""结构层：把 jaxpr 规范化，并在此基础上做结构比较。

本包是最底层，不依赖 algebra 或 functions。各模块分工：
- canonical：BFS 编号、依赖深度、区域同步置换，以及 canonicalize / compare_jaxprs；
- semantics：参数、抽象值和常量的精确编码；
- params：比较与代数适配共用的参数视图；
- support：按身份发现公开 lax 原语和包装函数产生的内部原语；
- unrooted：给没有输出根的纯计算也排出规范顺序；
- correspondence：构造并独立复核原图对应证据；
- derivatives：物化注册的一阶自定义求导规则；
- _typing：本包共用的类型别名与协议。
"""
