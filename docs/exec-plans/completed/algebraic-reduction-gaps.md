# 重构标量索引 IR 并补齐代数等价

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

状态：已完成。2026-10-14 用户要求补齐四类已确认的漏认，专注代数比较；结构比较保持现状，受保护文档不修改。

## 目标与范围

- `sum(x)` 与 `sum(x.reshape(-1))`。
- `sum(x)` 与 `sum(x[:2]) + sum(x[2:])`，推广到合法静态连续分块，保留覆盖与重数。
- `scan` 的加法累加与对应的 `sum`，保留初值、轴、各个 carry 和返回值。
- `abs(x)` 与 `where(x > 0, x, -x)`，严格区别错误条件和 JAX 在零点的求导约定。

继续采用理想实数语义和浮点警告；不将浮点执行差异误当作实数反例。只处理已成功追踪、语义已校验的 jaxpr，不分析一般 Python 源码。未满足规则条件时保留原边界或返回 unknown。

## 方案与取舍

具体设计见[标量索引 IR 设计](../../design-docs/archive/algebraic-reduction-gaps.md)。沿用索引 DAG、精确系数、符号有限求和和独立证据校验。

以带索引的标量语义为共同中间表示。reshape/切片/转置/广播转换为符号整数索引映射；条件与非线性算子使用共享索引环境的标量调用；sum/matmul/累加递推使用有限求和。规范化阶段以带见证的域双射、区间差分和多项式规则统一处理组合，不依赖原算子来源。

## 工作项

- [x] 保存范围、方案、拒绝条件及验收计划。
- [x] 重构符号索引、标量函数作用域和求和域证据，接入 reshape/切片及独立复核。
- [x] 实现已审查的 scan 累加转换，验证反馈与返回值。
- [x] 实现 abs/where 互认，保留零点导数差异。
- [x] 补齐正例、空/单位轴、错误范围/反馈/条件、预算与证据篡改测试。
- [x] 运行全量旧测试及新增测试、Ruff、格式、示例与文档检查，审查差异后归档。

## 验证要求

每类规则必须通过实际 JAX 追踪和底层代数证明复核。覆盖 float32、float16、bfloat16；求和覆盖标量、空域、单位域、多维及大维度符号预算。分块反例包含遗漏、重叠、重复片段、错误输入和步长；scan 反例包含错误初值、减法反馈、carry 依赖和可观察的历史输出；where 反例包含错误阈值、分支、输入和零点求导差异。能适用的 JVP/VJP 分别检查，不能从前向证明推断梯度等价。

最终执行 `uv run pytest`、`ruff check .`、`ruff format --check .`、tracing/代数示例、`tools/check_docs.py --changed` 和 `git diff --check`，使用锁定离线环境及 `/tmp/jax-reorder-uv-cache`。

## 决策与进展

2026-10-14：用户将目标从四类漏认的补充升级为核心 IR 重构。取消局部视图来源追溯，改用符号整数坐标、共享索引环境的标量调用、FiniteRange 域与 Recurrence 递推；matmul、sum 和独立累加进入同一标量公式。此前 scan 与 abs/where 适配迁移到此表示。

域规范化基于 IR 读坐标选择混合进制合并/拆分和平移，记录分支位置与具体见证。证明格式为 version=2；校验器检查原图来源、每个域指令、所有展开项和最终有序结果，不调用域规划或索引搜索。整数与域检查内核以独立小域解释器验证。

审查补齐了两个边界：单元素布尔广播必须保持来源坐标，不能把 `where(x[:1] > 0, x, -x)` 误认成 abs；前缀展开不能用于部分函数或带危险负偏移/负系数的坐标。scan 的 weak_type 变化可以保留实际输出契约后解释，同 dtype 转换不改变元素值；不同 dtype 转换仍显式存在。

## 验证结果

使用 Python 3.12、锁定 JAX/jaxlib 0.11.2，命令统一前缀为 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline`。

| 检查 | 最终结果 |
| --- | --- |
| `pytest -q` | **1775 passed，1 warning，424.87 秒**；原有 1600 项全部保留，新增 175 项 |
| `ruff check .` | 通过 |
| `ruff format --check .` | 126 个 Python 文件全部通过 |
| `python tests/inspect_jaxpr.py` | tracing 与结构比较通过 |
| `python tests/bfs_reordering.py` | 原有示例正反例通过 |
| `python tests/algebraic_equivalence.py` | 分配律和矩阵/乘加归约的 forward/JVP/VJP、独立证明与 CPU 编译通过 |
| `python tests/lax_inventory.py` | 锁定原语清单一致 |
| `python tests/adapter_inventory.py` | IR 字段与参数快照一致 |
| `python tests/collective_jaxpr.py --devices 2` / `--devices 4` | 各 22 个组合通过 |
| `python tools/check_docs.py --changed` / `git diff --check` | 通过 |

唯一 warning 是保留的上游 Pallas reciprocal 弃用提示。此前首轮全量为 1774 passed，补充广播反例及最终类型整理后，已重新运行上表所列最终全量。

新增测试覆盖：三种浮点 dtype 的完整 reshape 归约、连续和多维分块、空域/单元素块、先展平再分块、逐点函数跨形状映射、递推求解及三种模式、绝对值零点导数、错误覆盖/输入/返回顺序、独立作用域解释、证据篡改和百万长度符号预算。数值解释器只作为语义内核的验证证据，不替代函数等价证明。

## 保留边界

本轮完成上述四组需求及核心重构，不宣称任意数学恒等式的完备判定。动态索引、任意稀疏/步长求和域、一般递推、未审核原语和跨区域改写继续返回 unknown 或保留完整边界；见 [ALG-001/002](../index.md#待处理)。严格 `> 0` 的 where 与 abs 只证明前向等价，不能由此前向证明推导零点 JVP/VJP 相同。

受保护文档按用户要求保持原状。验收完成后，用户已授权提交并推送本轮改动。
