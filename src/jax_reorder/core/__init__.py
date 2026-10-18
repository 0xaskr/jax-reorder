"""结构层：jaxpr 规范化、语义编码与结构比较。

本包属于底层，不依赖 algebra 或 functions；模块职责：
- canonical：BFS 编号、依赖深度、区域同步置换与 compare_jaxprs。
- semantics：参数、抽象值与常量的精确编码。
- support：公开 lax 原语及包装辅助原语的身份发现。
- unrooted：未使用部分的完整图搜索。
- correspondence：原始图对应证据的构造与独立复核。
- derivatives：注册的一阶自定义求导规则物化。
"""
