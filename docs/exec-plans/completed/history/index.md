# 结构与代数的历史实施记录

以下保留原阶段的实现、审查、失败、实验和验收结果。记录已归档不表示后续缺陷已修复，
也不把旧测试数字当成当前源码的全量验收。现行规则见[设计索引](../../../design-docs/index.md)。

| 记录 | 内容 | 当前设计 |
| --- | --- | --- |
| [BFS 首轮审查与验收](bfs-reordering-validation.md) | 2026-10-11 的实际修复和 516 项测试等证据 | [完整图](../../../design-docs/current/complete-graph.md)、[原图对应](../../../design-docs/current/original-correspondence.md) |
| [cond 首轮修复与四层验证](cond-bindings-validation.md) | 闭包绑定复现、首轮验收与固定四层案例 | [区域绑定](../../../design-docs/current/region-bindings.md) |
| [代数后端实验与验收](algebraic-equivalence-validation.md) | 旧证明器/CAS 阶段、可信 SymPy 迁移、规模实验及修复记录 | [可信 SymPy](../../../design-docs/current/algebraic-equivalence.md) |

旧算法及接口方案另见[已被替代的设计](../../../design-docs/index.md#已被替代的设计)。
